import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from dutils import *
import wandb

# Define the number of samples M and the number of clusters
M = 10000
num_clusters = 4
samples_per_cluster = M // num_clusters
use_z = False
device = 'cuda'

# Centers (mu) and standard deviations (sigma)
centers = np.array([(0.0, 0.7), (0.7, 0.0), (0.0, -0.7), (-0.7, 0.0)])
zs = [torch.randn(5) for _ in range(len(centers))]
sigma = (0.08, 0.08)
rewards = [10.0, 7.5, 5.0, 2.5]

# Generate samples from Gaussian distributions
samples = []
aaa = []
bbb = []
colors = ['red', 'green', 'blue', 'purple']
for center, color, reward, z in zip(centers, colors, rewards, zs):
    cluster_samples = np.random.normal(loc=center, scale=sigma, size=(samples_per_cluster, 2))
    samples.append(cluster_samples)
    samples_reward = compute_reward(cluster_samples, center, reward)
    aaa.append(samples_reward)
    bbb.append(z.unsqueeze(0).repeat(samples_per_cluster, 1))

# Clip samples to ensure they remain in the range [-1, 1]^2
samples = np.concatenate(samples)
samples = np.clip(samples, -1, 1)
rewards = np.concatenate(aaa)
z_embed = torch.cat(bbb, dim=0)
if use_z:
    state = z_embed.to(device)
else:
    state = torch.zeros_like(z_embed).to(device)
action = torch.tensor(samples, dtype=torch.float32).to(device)
reward = torch.tensor(rewards, dtype=torch.float32).view(-1, 1).to(device)

data_sampler = Data_Sampler(state, action, reward, device)

state_dim = 5
action_dim = 2
max_action = 1.0

discount = 0.99
tau = 0.005
model_type = 'MLP'

T = 50
beta_schedule = 'vp'
hidden_dim = 128
lr = 3e-4

num_epochs = 1000
batch_size = 100
iterations = int(M / batch_size)
num_eval = 1000
agent = QL_Diffusion(state_dim=state_dim,
                     action_dim=action_dim,
                     max_action=max_action,
                     device=device,
                     discount=discount,
                     tau=tau,
                     beta_schedule=beta_schedule,
                     n_timesteps=T,
                     model_type=model_type,
                     hidden_dim=hidden_dim,
                     lr=lr,
                     eta=10.0)
# weight = torch.load('/home/supersglzc/code/Diffusion-Policies-for-Offline-RL/Qnet-z.pth')
# zs = weight['z']
# agent.critic.load_state_dict(weight['Q'])
# Qnet.load_state_dict(weight)
wandb_run = wandb.init(project='2d_bandit', mode='online', name=f'cql_wo_policyQ')
for i in range(num_epochs):
    agent.train(data_sampler, iterations=iterations, batch_size=batch_size, zs=zs)
    
    if i % 100 == 0:
        print("Epoch", i)
        fig, ax = plt.subplots()
        new_state = torch.zeros((num_eval, 5)).to(device)
        # new_action = agent.actor.sample(new_state)
        new_action = agent.actor(new_state)
        new_action = new_action.detach().cpu().numpy()
        ax.scatter(new_action[:, 0], new_action[:, 1])
        ax.set_xlim((-1, 1))
        ax.set_ylim((-1, 1))
        fig.canvas.draw()  # Draw the canvas, cache the renderer
        image_flat = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
        img = image_flat.reshape(*reversed(fig.canvas.get_width_height()), 3)

        fig, ax = plt.subplots()
        ax.scatter(data_sampler.target_action[:, 0].cpu().numpy(), data_sampler.target_action[:, 1].cpu().numpy(), alpha=0.5)
        ax.set_xlim((-1, 1))
        ax.set_ylim((-1, 1))
        fig.canvas.draw()  # Draw the canvas, cache the renderer
        image_flat = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
        target_img = image_flat.reshape(*reversed(fig.canvas.get_width_height()), 3)
        img = np.concatenate([img, target_img], axis=1)
        plt.close()
        img = wandb.Image(img)
        q_value = generate_q_map(agent.critic, if_double=True)
        q_value = wandb.Image(q_value)
        wandb.log({
            'images/actions': img,
            'images/q_value': q_value,}, 
            step=i)
        

        # num_points = 400
        # x, y = np.meshgrid(np.linspace(-1, 1, num_points), np.linspace(-1, 1, num_points))
        # grid = np.stack([x, y], axis=-1)
        # samples = torch.tensor(grid, dtype=torch.float32).reshape(num_points*num_points, 2)
        # zs_grid = []
        # samples_grid = []
        # for i in range(samples.shape[0]):
        #     x1 = samples[i][0].item()
        #     y2 = samples[i][1].item()
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
        #     samples_grid.append([x1, y2])
        # zs_grid = torch.stack(zs_grid).to(device)
        # samples_grid = torch.tensor(samples_grid).to(device)
        # if use_z:
        #     Q_values1, Q_values2 = agent.critic(zs_grid, samples_grid)
        #     Q_values1 = Q_values1.detach().cpu().reshape(num_points, num_points).numpy()
        #     Q_values2 = Q_values2.detach().cpu().reshape(num_points, num_points).numpy()
        # plt.figure(figsize=(8, 6))
        # plt.contourf(x, y, Q_values1, levels=50, cmap='viridis')  # Use a colormap that shows good gradient changes
        # plt.colorbar(label='Q-value')
        # plt.title('Q-value Heatmap of 2D Actions')
        # plt.xlabel('$a_1$ (Action Dimension 1)')
        # plt.ylabel('$a_2$ (Action Dimension 2)')
        # plt.show()
        # plt.figure(figsize=(8, 6))
        # plt.contourf(x, y, Q_values2, levels=50, cmap='viridis')  # Use a colormap that shows good gradient changes
        # plt.colorbar(label='Q-value')
        # plt.title('Q-value Heatmap of 2D Actions')
        # plt.xlabel('$a_1$ (Action Dimension 1)')
        # plt.ylabel('$a_2$ (Action Dimension 2)')
        # plt.show()

        # torch.save({
        #     'Q': agent.critic.state_dict(),
        #     'z': zs
        # }, '/home/supersglzc/code/Diffusion-Policies-for-Offline-RL/Qnet-z.pth')

        
