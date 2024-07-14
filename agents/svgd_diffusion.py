# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.logger import logger
import wandb
from agents.diffusion import Diffusion
from agents.model import MLP
from agents.helpers import EMA


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


class Diffusion_SVGD(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 max_q_backup=False,
                 eta=1.0,
                 beta_schedule='linear',
                 n_timesteps=100,
                 ema_decay=0.995,
                 step_start_ema=1000,
                 update_ema_every=5,
                 lr=3e-4,
                 lr_decay=False,
                 lr_maxt=1000,
                 grad_norm=1.0,
                 num_steps_per_epoch=1000,
                 ):

        # self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)

        # self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
        #                        beta_schedule=beta_schedule, n_timesteps=n_timesteps,).to(device)
        from pql.models.diffusion_mlp import DiffusionPolicy
        self.actor = DiffusionPolicy(state_dim, action_dim, n_timesteps, device=device).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.lr_decay = lr_decay
        self.grad_norm = grad_norm
        self.num_steps_per_epoch = num_steps_per_epoch

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=0.)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.eta = eta  # q_learning weight
        self.device = device
        self.max_q_backup = max_q_backup

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

    def train(self, replay_buffer, iterations, batch_size=256, log_writer=None):

        metric = {'q_loss': [], 'cql_loss': [], 'actor_loss': []}
        for _ in range(iterations):
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            """ Q Training """
            current_q1, current_q2 = self.critic(state, action)

            if self.max_q_backup:
                next_state_rpt = torch.repeat_interleave(next_state, repeats=10, dim=0)
                next_action_rpt = self.ema_model(next_state_rpt)
                target_q1, target_q2 = self.critic_target(next_state_rpt, next_action_rpt)
                target_q1 = target_q1.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q2 = target_q2.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q = torch.min(target_q1, target_q2)
            else:
                next_action = self.ema_model(next_state)
                target_q1, target_q2 = self.critic_target(next_state, next_action)
                target_q = torch.min(target_q1, target_q2)

            target_q = (reward + not_done * self.discount * target_q).detach()

            # CQL
            cql_temp = 1.0
            cql_weight = 2.0
            num_repeat = 10
            batch_size = state.shape[0]
            action_dim = action.shape[-1]
            random_actions = torch.FloatTensor(batch_size * num_repeat, action_dim).uniform_(-1, 1).to(state.device)
            temp_states = state.unsqueeze(1).repeat(1, num_repeat, 1).view(state.shape[0] * num_repeat, state.shape[1])
            current_actions = self.ema_model(temp_states).detach()

            random_values1, random_values2 = self.critic(temp_states, random_actions)
            current_pi_values1, current_pi_values2 = self.critic(temp_states, current_actions)
            random_values1 = random_values1.reshape(state.shape[0], num_repeat, 1)
            random_values2 = random_values2.reshape(state.shape[0], num_repeat, 1)
            current_pi_values1 = current_pi_values1.reshape(state.shape[0], num_repeat, 1)
            current_pi_values2 = current_pi_values2.reshape(state.shape[0], num_repeat, 1)

            cat_q1 = torch.cat([random_values1, current_pi_values1], 1)
            cat_q2 = torch.cat([random_values2, current_pi_values2], 1)

            cql1_scaled_loss = ((torch.logsumexp(cat_q1 / cql_temp, dim=1).mean() * cql_temp) - current_q1.mean()) * cql_weight
            cql2_scaled_loss = ((torch.logsumexp(cat_q2 / cql_temp, dim=1).mean() * cql_temp) - current_q2.mean()) * cql_weight

            q_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            cql_loss = cql1_scaled_loss + cql2_scaled_loss
            critic_loss = q_loss + cql_loss

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.critic_optimizer.step()

            """SVGD"""
            from SVGD import SVGD_algo
            n_iter = 500
            num_particles = 4096
            kernel_length = 0.0005
            # randomly generate actions that covers the entire action space
            if num_particles % state.shape[0] != 0:
                raise ValueError('SVGD init number of particles must be divisible by batch size!')
            state_rpt = torch.repeat_interleave(state, repeats=num_particles // state.shape[0], dim=0)
            particles = self.max_action * (torch.rand(num_particles, self.action_dim, device=state.device) * 2 - 1)
            target_action = SVGD_algo().update(state_rpt, particles, self.critic, num_particles, n_iter=n_iter, stepsize=0.01, h=kernel_length)

            """ Policy Training """
            actor_loss = self.actor.get_loss(state_rpt, target_action.type(torch.float32).to(self.device))
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0: 
                actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.actor_optimizer.step()


            """ Step Target network """
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1

            """ Log """
            if self.step % self.num_steps_per_epoch == 0:
                wandb.log({
                    'Q Loss': q_loss.item(),
                    'CQL Loss': cql_loss.item(),
                    'Actor Loss': actor_loss.item(),
                    'Actor Grad Norm': actor_grad_norms.max().item(),
                    'Critic Grad Norm': critic_grad_norms.max().item()
                }, step=self.step // self.num_steps_per_epoch)

            metric['q_loss'].append(q_loss.item())
            metric['cql_loss'].append(cql_loss.item())
            metric['actor_loss'].append(actor_loss.item())
            
        if self.lr_decay: 
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        return metric

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        with torch.no_grad():
            action = self.actor(state_rpt)
            q_value = self.critic_target.q_min(state_rpt, action).flatten()
            idx = torch.multinomial(F.softmax(q_value), 1)
        return action[idx].cpu().data.numpy().flatten()

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic.pth'))


