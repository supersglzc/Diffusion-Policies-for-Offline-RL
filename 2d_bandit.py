import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from dutils import *

# Define the number of samples M and the number of clusters
M = 10000
num_clusters = 4
samples_per_cluster = M // num_clusters
use_z = True

# Centers (mu) and standard deviations (sigma)
centers = np.array([(0.0, 0.8), (0.8, 0.0), (0.0, -0.8), (-0.8, 0.0)])
zs = [torch.randn(5) for _ in range(len(centers))]
sigma = (0.1, 0.1)
rewards = [6.0, 4.0, 2.5, 1.0]
reward_sigma = 0.05

# Generate samples from Gaussian distributions
samples = np.empty((0, 2), dtype=np.float32)
aaa = []
bbb = []
colors = ['red', 'green', 'blue', 'purple']
for center, color, reward, z in zip(centers, colors, rewards, zs):
    cluster_samples = np.random.normal(loc=center, scale=sigma, size=(samples_per_cluster, 2))
    samples = np.vstack((samples, cluster_samples))
    samples_reward = compute_reward(cluster_samples, center, reward)
    aaa.append(samples_reward)
    bbb.append(z.unsqueeze(0).repeat(samples_per_cluster, 1))
    plt.scatter(cluster_samples[:, 0], cluster_samples[:, 1], color=color, alpha=0.5, label=f'Center {center}, r {reward}')
# Plotting
plt.title('Scatter Plot of Actions from Gaussian Distributions')
plt.xlabel('Action Dimension 1')
plt.ylabel('Action Dimension 2')
plt.legend()
plt.show()

# Clip samples to ensure they remain in the range [-1, 1]^2
samples = np.clip(samples, -1, 1)
rewards = np.concatenate(aaa)
z_embed = torch.cat(bbb, dim=0)

if use_z:
    Qnet = MLPNet(in_dim=2+5, out_dim=1, hidden_layers=[256, 256, 128])
else:
    Qnet = MLPNet(in_dim=2, out_dim=1)
Qnet_optimizer = torch.optim.AdamW(Qnet.parameters(), 0.0005)

batch_size = 500
for iter in range(2000):
    prev = 0
    losses = []
    for j in range(M // batch_size):
        b_a = torch.tensor(samples[prev:prev+batch_size], dtype=torch.float32)
        b_z = z_embed[prev:prev+batch_size]
        b_input = torch.cat([b_a, b_z], dim=1)
        b_r = torch.tensor(rewards[prev:prev+batch_size], dtype=torch.float32)
        if use_z:
            pred = Qnet(b_input)
        else:
            pred = Qnet(b_a)
        loss = nn.functional.mse_loss(pred, b_r)
        grad_norm = optimizer_update(Qnet_optimizer, loss)
        losses.append(loss.item())
        prev += batch_size
    print(iter, np.array(losses).mean())
# torch.save(Qnet.state_dict(), '/home/supersglzc/code/Diffusion-Policies-for-Offline-RL/Qnet-z.pth')

# weight = torch.load('/home/supersglzc/code/Diffusion-Policies-for-Offline-RL/Qnet-z.pth')
# Qnet.load_state_dict(weight)

# Generate a grid of points
num_points = 33
x, y = np.meshgrid(np.linspace(-1, 1, num_points), np.linspace(-1, 1, num_points))
grid = np.stack([x, y], axis=-1)
samples = torch.tensor(grid, dtype=torch.float32).reshape(num_points*num_points, 2)
zs_grid = []
samples_grid = []
for i in range(samples.shape[0]):
    x1 = samples[i][0].item()
    y2 = samples[i][1].item()
    if y2 > x1 and y2 > -x1:
        idx = 0
    elif y2 < x1 and y2 < -x1:
        idx = 2
    elif y2 < x1 and y2 > -x1:
        idx = 1
    elif y2 > x1 and y2 < -x1:
        idx = 3
    else:
        continue
    zs_grid.append(zs[idx])
    samples_grid.append([x1, y2])
zs_grid = torch.stack(zs_grid)
samples_grid = torch.tensor(samples_grid)

action_optimizer = torch.optim.Adam([samples_grid], lr=0.01, eps=1e-5)
samples_grid.requires_grad_(True)
Q = Qnet(torch.cat([samples_grid, zs_grid], dim=1))
# Q = Qnet(samples_grid)
loss = -Q.mean()
action_optimizer.zero_grad(set_to_none=True)
loss.backward()
gradients = -samples_grid.grad
samples_grid.requires_grad_(False)
# action_optimizer.step()
# gradients = gradients.reshape(num_points, num_points, 2).numpy()

plt.figure(figsize=(10, 8))
plt.quiver(samples_grid[:, 0].numpy(), samples_grid[:, 1].numpy(), gradients[:,0].numpy(), gradients[:,1].numpy(), color='r')
plt.title('Vector Field')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.axis('equal')
plt.show()

num_points = 400
x, y = np.meshgrid(np.linspace(-1, 1, num_points), np.linspace(-1, 1, num_points))
grid = np.stack([x, y], axis=-1)
samples = torch.tensor(grid, dtype=torch.float32).reshape(num_points*num_points, 2)
zs_grid = []
samples_grid = []
for i in range(samples.shape[0]):
    x1 = samples[i][0].item()
    y2 = samples[i][1].item()
    if y2 > x1 and y2 > -x1:
        idx = 0
    elif y2 < x1 and y2 < -x1:
        idx = 2
    elif y2 < x1 and y2 > -x1:
        idx = 1
    elif y2 > x1 and y2 < -x1:
        idx = 3
    elif y2 == x1 and y2 == -x1:  # Exactly at the origin, rare case
        idx = random.choice([0, 1, 2, 3])
    elif y2 == x1:  # On the line y = x
        idx = random.choice([0, 1])
    elif y2 == -x1:  # On the line y = -x
        idx = random.choice([2, 3])
    zs_grid.append(zs[idx])
    samples_grid.append([x1, y2])
zs_grid = torch.stack(zs_grid)
samples_grid = torch.tensor(samples_grid)
Q_values = Qnet(torch.cat([samples_grid, zs_grid], dim=1)).detach().reshape(num_points, num_points).numpy()
plt.figure(figsize=(8, 6))
plt.contourf(x, y, Q_values, levels=50, cmap='viridis')  # Use a colormap that shows good gradient changes
plt.colorbar(label='Q-value')
plt.title('Q-value Heatmap of 2D Actions')
plt.xlabel('$a_1$ (Action Dimension 1)')
plt.ylabel('$a_2$ (Action Dimension 2)')
plt.show()