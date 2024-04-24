import gym
import numpy as np
import os
import torch
import d4rl
from utils.traj_data_sampler import Traj_Data_Sampler
from torchrl.data.datasets.d4rl import D4RLExperienceReplay
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

env_name = 'antmaze-large-diverse-v0'
t = gym.make(env_name)
dataset = t.get_dataset()
if dataset["terminals"][-1] == False:
    dataset["timeouts"][-1] = True
assert "observations_next" not in dataset and "next_observations" not in dataset 
assert dataset["timeouts"].shape[0] == dataset["observations"].shape[0]
data = {}
if env_name == "antmaze-medium-play-v2":
    # solve the little bug within this dataset
    rewid = np.where(dataset["rewards"]>0.001)[0]
    positions = dataset["observations"][rewid, :2]
    badid = rewid[~np.all(positions > 19.0, axis=1)]
    print("{} badid detected".format(badid.shape[0]))
    dataset["rewards"][badid] = 0.0
    dataset["terminals"][badid] = 0
# assert set(np.where(np.abs(dataset["observations"][1:,0] - dataset["observations"][:-1,0]) > 1.)[0]).issubset(set(np.where(dataset["timeouts"])[0]))
assert np.all(np.where(dataset["rewards"])[0] == np.where(dataset["terminals"])[0])
doneid = dataset["terminals"] | dataset["timeouts"]
start_id = np.where(doneid)[0]+1
assert start_id[-1] == doneid.shape[0]
assert start_id[0] != 0
start_id = [0] + [i for i in start_id]
data = {"states":[], "next_states":[], "done":[], "is_finished":[], "rewards":[], "actions":[]}
for i in range(len(start_id) - 1):
    if start_id[i+1] - start_id[i] < 5:
        continue
    if dataset["terminals"][start_id[i+1]-1]:
        data["states"].append(dataset["observations"][start_id[i]: start_id[i+1]])
        next_states = np.zeros_like(data["states"][-1])
        next_states[:-1] = data["states"][-1][1:]
        data["next_states"].append(next_states)
        data["actions"].append(dataset["actions"][start_id[i]: start_id[i+1]])
        data["rewards"].append(dataset["rewards"][start_id[i]: start_id[i+1], None])
        done = np.zeros((data["states"][-1].shape[0], 1))
        done[-1, 0] = 1
        data["done"].append(done)
        data["is_finished"].append(done)
    elif dataset["timeouts"][start_id[i+1]-1]:
        data["states"].append(dataset["observations"][start_id[i]: start_id[i+1]-1])
        data["next_states"].append(dataset["observations"][start_id[i]+1: start_id[i+1]])
        data["actions"].append(dataset["actions"][start_id[i]: start_id[i+1]-1])
        data["rewards"].append(dataset["rewards"][start_id[i]: start_id[i+1]-1, None])
        done = np.zeros((data["states"][-1].shape[0], 1))
        done[-1, 0] = 1
        data["done"].append(done)
        data["is_finished"].append(np.zeros_like(data["rewards"][-1]))
    else:
        assert False
for k in ["states", "next_states", "done", "is_finished", "rewards", "actions"]:
    data[k] = np.concatenate(data[k])
    size = data[k].shape[0]
print("data num {}".format(size))
for k in ["states", "next_states", "done", "is_finished", "rewards", "actions"]:
    assert data[k].shape[0] == size
    assert data[k].ndim == 2
    # bootstrap by 0 ignore is_finished
data["returns"] = np.zeros((data["states"].shape[0], 1))
last = 0
for i in range(data["returns"].shape[0] - 1, -1, -1):
    last = data["rewards"][i, 0] + 0.99 * last * (1. - data["done"][i, 0])
    data["returns"][i, 0] = last
observations = data['states']
actions = data['actions']
rewards = data['rewards']
terminals = data['done']

dataset, action_dataset = [], []
states, temp_actions = [], []
s = 0

env_kwargs = t.env.env.spec.kwargs
for obs, action, reward, terminal in zip(observations, actions, rewards, terminals):
    states.append(torch.tensor(obs))
    temp_actions.append(torch.tensor(action))
    if terminal:
        if len(states) == 998:
            states = torch.stack(states)
            temp_actions = torch.stack(temp_actions)
            dataset.append(states)
            action_dataset.append(temp_actions)
        states, temp_actions = [], []
    s += 1
print(len(dataset))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import wandb
import umap
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
# from torchvision.utils import make_grid
from pyclustering.cluster import cluster_visualizer_multidim
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import SIMPLE_SAMPLES
from vae import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 250
num_training_updates = 3000


embedding_dim = 64
num_embeddings = 128
H_dim = 10
K_dim = 10
commitment_cost = 0.25

decay = 0.99

learning_rate = 1e-3

wandb_run = wandb.init(project='vqvae', mode='online', name=f'vqvae-H-10')

# dataset = torch.load('/home/supersglzc/code/o2o_diffusion/data/antmaze-v3-4000traj-states.pth')
# action_dataset = torch.load('/home/supersglzc/code/o2o_diffusion/data/antmaze-v3-4000traj-actions.pth')
# def downsample(target_len, state):
#     indices = torch.linspace(0, state.shape[0]-1, steps=int(target_len)).long()
#     return state[indices]
new_dataset = []
new_action_dataset = []
train_dataset, validate_dataset, train_action_dataset, validate_action_dataset = [], [], [], []
plot_dataset = []
for i in range(len(dataset)):
    # traj = downsample(target_len=100, state=dataset[i][:, :2])
    # actions = downsample(target_len=100, state=action_dataset[i])
    traj = dataset[i]  # [:, :2]
    actions = action_dataset[i]
    from pql.utils.intrinsic import get_embedder
    embedder, _ = get_embedder(10, inputs_dim=2)
    pos = embedder(traj[:, :2])
    embed_traj = torch.cat([pos, traj[:, 2:]], dim=1)
    # new_dataset.append(traj)
    # new_action_dataset.append(actions)
    # plot_dataset.append(traj[:, :2].cpu())
    train_dataset.append(embed_traj)
    train_action_dataset.append(actions)
    validate_dataset.append(traj)
    validate_action_dataset.append(actions)
    plot_dataset.append(traj[:, :2].cpu())

state_dim = train_dataset[0].shape[-1]
action_dim = train_action_dataset[0].shape[-1]
model = Model(state_dim, action_dim, H_dim, num_embeddings, embedding_dim, 
              commitment_cost).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

# model.train()
train_res_recon_error = []
train_res_perplexity = []
train_res_distance_loss = []
train_res_max_loss = []
for i in range(num_training_updates):
    prev = 0
    for j in range(len(train_dataset) // batch_size):
        tau, tau_action, state_chunk = [], [], []
        for b in range(batch_size):
            cur_tau = train_dataset[prev+b].to(device)
            cur_tau_action = train_action_dataset[prev+b].to(device)
            indices = torch.randint(cur_tau.shape[0]-K_dim, size=(1,), device=device)
            tau.append(cur_tau[indices:indices+K_dim].unsqueeze(0))
            tau_action.append(cur_tau_action[indices:indices+K_dim].unsqueeze(0))
            indices = torch.randint(cur_tau.shape[0]-H_dim-1, size=(1,), device=device)
            state_chunk.append(cur_tau[indices:indices+H_dim+1].unsqueeze(0))
        tau = torch.cat(tau, dim=0)
        tau_action = torch.cat(tau_action, dim=0)
        # indices = torch.randint(100-H_dim-1, size=(batch_size,), device=device)
        # state_chunk = []
        # for s in range(H_dim + 1):
        #     state_chunk.append(tau[torch.arange(batch_size), indices + s].unsqueeze(1))
        state_chunk = torch.cat(state_chunk, dim=0)
        optimizer.zero_grad()
        loss, result_dict = model(tau, tau_action, state_chunk)
        loss.backward()
        optimizer.step()
        
        train_res_recon_error.append(result_dict['reconstruction_loss'].item())
        train_res_perplexity.append(result_dict['perplexity'].item())
        train_res_distance_loss.append(result_dict['distance_loss'].item())
        train_res_max_loss.append(result_dict['max_loss'].item())
        prev += batch_size

    print('%d iterations' % (i+1))
    print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
    print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
    wandb.log({"recon_error": np.mean(train_res_recon_error[-100:]),
               'perplexity': np.mean(train_res_perplexity[-100:]),
               'distance_loss': np.mean(train_res_distance_loss[-100:]),
               'max_loss': np.mean(train_res_max_loss[-100:]),
               }, step=i)
    
    if i % 100 == 0:
        # project embedding
        # model.eval()
        projection, embedding_img, xlim, ylim = get_projection(model._vq_vae._embedding.weight.data.cpu())
        # embedding_img = wandb.Image(embedding_img)
        # clusters
        clusters, centers = get_clusters(projection)
        # clusters = [[i] for i in range(embedding_dim)]
        # evaluate validation dataset
        tau, tau_action = [], []
        pos = []
        encodes = []
        for _ in range(200):
            tau, tau_action = [], []
            for b in range(len(validate_dataset)):
                cur_tau = validate_dataset[b].to(device)
                cur_tau_action = validate_action_dataset[b].to(device)
                indices = torch.randint(cur_tau.shape[0]-K_dim, size=(1,), device=device)
                sss = cur_tau[indices:indices+K_dim]
                aaaa = embedder(sss[:, :2])
                embed_sss = torch.cat([aaaa, sss[:, 2:]], dim=1)
                tau.append(embed_sss.unsqueeze(0))
                pos.append(cur_tau[indices:indices+K_dim, :2].unsqueeze(0).cpu())
                tau_action.append(cur_tau_action[indices:indices+K_dim].unsqueeze(0))
            tau = torch.cat(tau, dim=0)
            tau_action = torch.cat(tau_action, dim=0)
            encodings = model.encode(tau, tau_action)
            encodes.append(encodings.cpu())
        encodings = torch.cat(encodes, dim=0)
        pos = torch.cat(pos, dim=0)
        # clusters = [[] for _ in range(num_embeddings)]
        # for k in range(encodings.shape[0]):
        #     idx = torch.where(encodings[k] == 1)[0]
        #     clusters[idx].append(k)
        # new_clusters = []
        # new_clusters_id = {}
        # new_projection = []
        # for k in range(num_embeddings):
        #     if len(clusters[k]) != 0:
        #         new_clusters.append(clusters[k])
        #         new_clusters_id[k] = projection[k]
        #         new_projection.append(projection[k])
        # for k in range(len(new_clusters)):
        #     img = plot_cluster(env_kwargs, tau[:, :2].cpu().numpy(), [new_clusters[k]])
        #     fig, ax = plt.subplots()
        #     key = list(new_clusters_id.keys())[k]
        #     ax.scatter(new_clusters_id[key][0], new_clusters_id[key][1], alpha=0.3)
        #     ax.set_xlim(xlim)
        #     ax.set_ylim(ylim)
        #     fig.canvas.draw()  # Draw the canvas, cache the renderer
        #     image_flat = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
        #     dd_img = image_flat.reshape(*reversed(fig.canvas.get_width_height()), 3)
        #     img = np.concatenate([img, dd_img], axis=1)
        #     img = wandb.Image(img)
        #     wandb.log({f'images/encoder cluster{k}': img})
        fig, ax = plt.subplots()
        colors = ['#808080', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', 
            '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', 
            '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', 
            '#000075', '#ffffff', '#000000']
        for i in range(len(clusters)):
            for j in range(len(clusters[i])):
                ax.scatter(projection[clusters[i][j],0], projection[clusters[i][j],1], alpha=0.3, c=colors[i])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        fig.canvas.draw()  # Draw the canvas, cache the renderer
        image_flat = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
        color_embedding_img = image_flat.reshape(*reversed(fig.canvas.get_width_height()), 3)
        embedding_img = np.concatenate([embedding_img, color_embedding_img], axis=1)
        embedding_img = wandb.Image(embedding_img)
        wandb.log({'images/embedding': embedding_img})
        traj_clusters = [[] for _ in range(len(clusters))]
        for k in range(encodings.shape[0]):
            idx = torch.where(encodings[k] == 1)[0].item()
            for g in range(len(clusters)):
                if idx in clusters[g]:
                    traj_clusters[g].append(k)
        for k in range(len(traj_clusters)):
            img = plot_cluster(env_kwargs, pos.numpy(), [traj_clusters[k]], c=colors[k])
            fig, ax = plt.subplots()
            ax.scatter(centers[k][0], centers[k][1], alpha=0.3, s=100)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            fig.canvas.draw()  # Draw the canvas, cache the renderer
            image_flat = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
            dd_img = image_flat.reshape(*reversed(fig.canvas.get_width_height()), 3)
            img = np.concatenate([img, dd_img], axis=1)
            img = wandb.Image(img)
            wandb.log({f'images/encoder cluster{k}': img})
        # img = plot_cluster(env_kwargs, plot_dataset, traj_clusters)
        # img = wandb.Image(img)
        # wandb.log({f'images/encoder cluster': img, 'images/embedding': embedding_img, 'images/num_clusters': len(traj_clusters)})
        
        # save model
        # path = f"{wandb_run.dir}/model.pth"
        # torch.save(model.state_dict(), path)  # save policy network in *.pth
        # model_artifact = wandb.Artifact(wandb_run.id, type="model", description=f"num clusters: {len(traj_clusters)}")
        # model_artifact.add_file(path)
        # wandb.save(path, base_path=wandb_run.dir)
        # wandb_run.log_artifact(model_artifact)

        # model.train()
