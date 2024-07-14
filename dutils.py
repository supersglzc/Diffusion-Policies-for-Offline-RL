import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from agents.helpers import EMA
from agents.helpers import SinusoidalPosEmb


def generate_q_map(Qnet, if_double=False, device='cuda'):
    num_points = 400
    x, y = np.meshgrid(np.linspace(-1, 1, num_points), np.linspace(-1, 1, num_points))
    grid = np.stack([x, y], axis=-1)
    samples = torch.tensor(grid, dtype=torch.float32).reshape(num_points*num_points, 2).to(device)
    states = torch.zeros((samples.shape[0], 5)).to(device)
    if if_double:
        Q1, Q2 = Qnet(states, samples)
        Q1 = Q1.detach().reshape(num_points, num_points).cpu().numpy()
        Q2 = Q2.detach().reshape(num_points, num_points).cpu().numpy()
        fig, ax = plt.subplots()
        img1 = ax.contourf(x, y, Q1, levels=50, cmap='viridis')  # Use a colormap that shows good gradient changes
        fig.colorbar(img1, label='Q-value')
        ax.set_title('Q1')
        ax.set_xlabel('$a_1$ (Action Dimension 1)')
        ax.set_ylabel('$a_2$ (Action Dimension 2)')
        fig.canvas.draw()  # Draw the canvas, cache the renderer
        image_flat = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
        img1 = image_flat.reshape(*reversed(fig.canvas.get_width_height()), 3)
        plt.close()
        fig, ax = plt.subplots()
        img2 = ax.contourf(x, y, Q2, levels=50, cmap='viridis')  # Use a colormap that shows good gradient changes
        fig.colorbar(img2, label='Q-value')
        ax.set_title('Q2')
        ax.set_xlabel('$a_1$ (Action Dimension 1)')
        ax.set_ylabel('$a_2$ (Action Dimension 2)')
        fig.canvas.draw()  # Draw the canvas, cache the renderer
        image_flat = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
        img2 = image_flat.reshape(*reversed(fig.canvas.get_width_height()), 3)
        plt.close()
        img = np.concatenate([img1, img2], axis=1)
    else:
        Q = Qnet(states, samples)
        Q_values = Q.detach().reshape(num_points, num_points).cpu().numpy()
        fig, ax = plt.subplots()
        fig.contourf(x, y, Q_values, levels=50, cmap='viridis')  # Use a colormap that shows good gradient changes
        fig.colorbar(label='Q-value')
        fig.title('Q')
        fig.xlabel('$a_1$ (Action Dimension 1)')
        fig.ylabel('$a_2$ (Action Dimension 2)')
        fig.canvas.draw()  # Draw the canvas, cache the renderer
        image_flat = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
        img = image_flat.reshape(*reversed(fig.canvas.get_width_height()), 3)
        plt.close()
    return img

def create_simple_mlp(in_dim, out_dim, hidden_layers, act=nn.ELU):
    layer_nums = [in_dim, *hidden_layers, out_dim]
    model = []
    for idx, (in_f, out_f) in enumerate(zip(layer_nums[:-1], layer_nums[1:])):
        model.append(nn.Linear(in_f, out_f))
        if idx < len(layer_nums) - 2:
            model.append(act())
    return nn.Sequential(*model)

def optimizer_update(optimizer, objective):
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        grad_norm = clip_grad_norm_(parameters=optimizer.param_groups[0]["params"],
                                        max_norm=1.)
        optimizer.step()
        return grad_norm

class MLPNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers=None):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [512, 256, 128]
        self.net = create_simple_mlp(in_dim=in_dim,
                                     out_dim=out_dim,
                                     hidden_layers=hidden_layers)

    def forward(self, x):
        return self.net(x)
    
def compute_reward(samples, center, max_reward, max_distance=1.0):
    distances = np.sqrt((samples[:, 0] - center[0]) ** 2 + (samples[:, 1] - center[1]) ** 2)
    rewards = max_reward * (1 - distances / max_distance)
    rewards = np.maximum(0, rewards)
    
    return rewards


class MLP(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 t_dim=4,
                 hidden_dim=32):

        super(MLP, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                       nn.Mish(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.Mish(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.Mish())

        self.final_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, time, state):

        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1)
        x = self.mid_layer(x)

        return self.final_layer(x)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class QL_Diffusion(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 max_q_backup=False,
                 eta=1.0,
                 model_type='MLP',
                 beta_schedule='linear',
                 n_timesteps=100,
                 ema_decay=0.995,
                 step_start_ema=1000,
                 update_ema_every=5,
                 lr=3e-4,
                 hidden_dim=32,
                 r_fun=None,
                 mode='whole_grad',
                 ):

        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device, hidden_dim=hidden_dim).to(device)
        from pql.models.diffusion_mlp import DiffusionPolicy
        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps,
                               ).to(device)
        # self.actor = DiffusionPolicy(state_dim, action_dim, n_timesteps, device=device).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        if r_fun is None:
            self.critic = Critic(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
            self.critic_target = copy.deepcopy(self.critic)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.eta = eta  # q_learning weight
        self.device = device
        self.max_q_backup = max_q_backup

        self.r_fun = r_fun
        self.mode = mode

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def train(self, replay_buffer, iterations, batch_size=100):

        for step in range(iterations):
            # Sample replay buffer / batch
            state, action, target_action, reward = replay_buffer.sample(batch_size)

            if self.r_fun is None:
                current_q1, current_q2 = self.critic(state, action)
                critic_loss = F.mse_loss(current_q1, reward) + F.mse_loss(current_q2, reward)
                
                # CQL
                cql_temp = 1.0
                cql_weight = 2.0
                num_repeat = 10
                batch_size = state.shape[0]
                action_dim = action.shape[-1]
                random_actions = torch.FloatTensor(batch_size * num_repeat, action_dim).uniform_(-1, 1).to(state.device)
                temp_states = state.unsqueeze(1).repeat(1, num_repeat, 1).view(state.shape[0] * num_repeat, state.shape[1])
                # current_actions = self.actor(temp_states).detach()

                random_values1, random_values2 = self.critic(temp_states, random_actions)
                # current_pi_values1, current_pi_values2 = self.critic(temp_states, current_actions)
                random_values1 = random_values1.reshape(state.shape[0], num_repeat, 1)
                random_values2 = random_values2.reshape(state.shape[0], num_repeat, 1)
                # current_pi_values1 = current_pi_values1.reshape(state.shape[0], num_repeat, 1)
                # current_pi_values2 = current_pi_values2.reshape(state.shape[0], num_repeat, 1)

                # cat_q1 = torch.cat([random_values1, current_pi_values1], 1)
                # cat_q2 = torch.cat([random_values2, current_pi_values2], 1)

                cat_q1 = random_values1
                cat_q2 = random_values2

                cql1_scaled_loss = ((torch.logsumexp(cat_q1 / cql_temp, dim=1).mean() * cql_temp) - current_q1.mean()) * cql_weight
                cql2_scaled_loss = ((torch.logsumexp(cat_q2 / cql_temp, dim=1).mean() * cql_temp) - current_q2.mean()) * cql_weight

                critic_loss = critic_loss + cql1_scaled_loss + cql2_scaled_loss

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
            
            # zs_grid = []
            # for i in range(target_action.shape[0]):
            #     x1 = target_action[i][0].item()
            #     y2 = target_action[i][1].item()
            #     if y2 > x1 and y2 > -x1:
            #         idx = 0
            #     elif y2 < x1 and y2 < -x1:
            #         idx = 2
            #     elif y2 < x1 and y2 > -x1:
            #         idx = 1
            #     elif y2 > x1 and y2 < -x1:
            #         idx = 3
            #     elif y2 == x1 and y2 == -x1:  # Exactly at the origin, rare case
            #         idx = random.choice([0, 1, 2, 3])
            #     elif y2 == x1:  # On the line y = x
            #         idx = random.choice([0, 1])
            #     elif y2 == -x1:  # On the line y = -x
            #         idx = random.choice([2, 3])
            #     zs_grid.append(zs[idx])
            # state = torch.stack(zs_grid).to(self.device)

            # self.critic.requires_grad_(False)
            # lim = 1 - 1e-5
            # target_action.clamp_(-lim, lim)
            # action_optimizer = torch.optim.Adam([target_action], lr=0.01, eps=1e-5)
            # for i in range(20):
            #     target_action.requires_grad_(True)
            #     Q = self.critic.q_min(state, target_action)
            #     loss = -Q.mean()
            #     optimizer_update(action_optimizer, loss)
            #     target_action.requires_grad_(False)
            #     target_action.clamp_(-lim, lim)
            # from copy import deepcopy
            # update = deepcopy(target_action.detach())
            # self.critic.requires_grad_(True)
            # replay_buffer.update(update)

            """ Policy Training """
            # a_state = torch.zeros_like(state).to(self.device)
            # # bc_loss = self.actor.loss(update, a_state)
            # bc_loss = self.actor.get_loss(a_state, update)

            bc_loss = self.actor.loss(action, state)
            if self.mode == 'whole_grad':
                new_action = self.actor(state)

            if self.r_fun is None:
                q1_new_action, q2_new_action = self.critic(state, new_action)
                q1_new_action = torch.exp(q1_new_action)
                q2_new_action = torch.exp(q2_new_action)
                if np.random.uniform() > 0.5:
                    lmbda = self.eta / q2_new_action.abs().mean().detach()
                    q_loss = - lmbda * q1_new_action.mean()
                else:
                    lmbda = self.eta / q1_new_action.abs().mean().detach()
                    q_loss = - lmbda * q2_new_action.mean()
            else:
                q_new_action = self.r_fun(new_action)
                lmbda = self.eta / q_new_action.abs().mean().detach()
                q_loss = - lmbda * q_new_action.mean()

            actor_loss = bc_loss + q_loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.actor.step_frozen()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            self.step += 1

        # Logging
        return bc_loss.item(), q_loss.item()

    def sample_action(self, state):
        # state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        # state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        # with torch.no_grad():
        #     action = self.actor.sample(state_rpt)
        #     q_value = self.critic_target.q_min(state_rpt, action).flatten()
        #     idx = torch.multinomial(F.softmax(q_value), 1)
        # return action[idx].cpu().data.numpy().flatten()
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            # action = self.actor.sample(state)
            action = self.actor(state)
        return action.cpu().data.numpy().flatten()

    def save_model(self, dir):
        torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
        torch.save(self.critic.state_dict(), f'{dir}/critic.pth')

    def load_model(self, dir):
        self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
        self.critic.load_state_dict(torch.load(f'{dir}/critic.pth'))


class Data_Sampler(object):
    def __init__(self, state, action, reward, device):

        self.state = state
        self.action = action
        self.target_action = self.action.clone()
        self.reward = reward

        self.size = self.state.shape[0]
        self.state_dim = self.state.shape[1]
        self.action_dim = self.action.shape[1]

        self.device = device

    def sample(self, batch_size):
        self.ind = torch.randint(0, self.size, size=(batch_size,))

        return (
            self.state[self.ind].to(self.device),
            self.action[self.ind].to(self.device),
            self.target_action[self.ind].to(self.device),
            self.reward[self.ind].to(self.device)
        )
    
    def update(self, new_action):
        self.target_action[self.ind] = new_action

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from agents.helpers import (cosine_beta_schedule,
                            linear_beta_schedule,
                            vp_beta_schedule,
                            extract,
                            Losses)
from utils.utils import Progress, Silent


class Diffusion(nn.Module):
    def __init__(self, state_dim, action_dim, model, max_action,
                 beta_schedule='linear', n_timesteps=100,
                 loss_type='l2', clip_denoised=True, predict_epsilon=True):
        super(Diffusion, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.model = model
        self.model_frozen = copy.deepcopy(self.model)

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(n_timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.loss_fn = Losses[loss_type]()

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, s, grad=True):
        if grad:
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, s))
        else:
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.model_frozen(x, t, s))

        if self.clip_denoised:
            x_recon.clamp_(-self.max_action, self.max_action)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    # @torch.no_grad()
    def p_sample(self, x, t, s, grad=True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s, grad=grad)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # @torch.no_grad()
    def p_sample_loop(self, state, shape, verbose=False, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        if return_diffusion: diffusion = [x]

        progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    # @torch.no_grad()
    def sample(self, state, *args, **kwargs):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        action = self.p_sample_loop(state, shape, *args, **kwargs)
        return action.clamp_(-self.max_action, self.max_action)

    def guided_sample(self, state, q_fun, start=0.2, verbose=False, return_diffusion=False):
        device = self.betas.device
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        x = torch.randn(shape, device=device)
        i_start = self.n_timesteps * start

        if return_diffusion: diffusion = [x]

        def guided_p_sample(x, t, s, fun):
            b, *_, device = *x.shape, x.device
            with torch.no_grad():
                model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s)
            noise = torch.randn_like(x)
            # no noise when t == 0
            nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

            # Involve Function Guidance
            a = model_mean.clone().requires_grad_(True)
            q_value = fun(s, a)
            # q_value = q_value / q_value.abs().mean().detach()  # normalize q
            grads = torch.autograd.grad(outputs=q_value, inputs=a, create_graph=True, only_inputs=True)[0].detach()
            return (model_mean + model_log_variance * grads) + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            if i <= i_start:
                x = guided_p_sample(x, timesteps, state, q_fun)
            else:
                with torch.no_grad():
                    x = self.p_sample(x, timesteps, state)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        x = x.clamp_(-self.max_action, self.max_action)

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1).clamp_(-self.max_action, self.max_action)
        else:
            return x

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, state, t, weights=1.0):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.model(x_noisy, t, state)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)

        return loss

    def loss(self, x, state, weights=1.0):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, state, t, weights)

    def forward(self, state, *args, **kwargs):
        return self.sample(state, *args, **kwargs)

    def step_frozen(self):
        for param, target_param in zip(self.model.parameters(), self.model_frozen.parameters()):
            target_param.data.copy_(param.data)

    def sample_t_middle(self, state):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        t = np.random.randint(0, int(self.n_timesteps*0.2))
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state, grad=(i == t))
        action = x
        return action.clamp_(-self.max_action, self.max_action)

    def sample_t_last(self, state):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        device = self.betas.device

        x = torch.randn(shape, device=device)
        cur_T = np.random.randint(int(self.n_timesteps * 0.8), self.n_timesteps)
        for i in reversed(range(0, cur_T)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            if i != 0:
                with torch.no_grad():
                    x = self.p_sample(x, timesteps, state)
            else:
                x = self.p_sample(x, timesteps, state)

        action = x
        return action.clamp_(-self.max_action, self.max_action)

    def sample_last_few(self, state):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        device = self.betas.device

        x = torch.randn(shape, device=device)
        nest_limit = 5
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            if i >= nest_limit:
                with torch.no_grad():
                    x = self.p_sample(x, timesteps, state)
            else:
                x = self.p_sample(x, timesteps, state)

        action = x
        return action.clamp_(-self.max_action, self.max_action)

