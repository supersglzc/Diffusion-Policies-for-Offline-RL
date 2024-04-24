import gym
import numpy as np
import os
import torch
import d4rl
from utils.traj_data_sampler import Traj_Data_Sampler
from torchrl.data.datasets.d4rl import D4RLExperienceReplay
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

env_name = 'antmaze-large-diverse-v0' # 'antmaze-large-diverse-v0'
# env = gym.make(env_name)
#  # load d4rl qlearning dataset
# dataset = d4rl.qlearning_dataset(env)
# # detect terminal states
# # terminal_state = np.where(np.all(dataset['next_observations'][:-1] != dataset['observations'][1:], axis = 1))[0]
# # dataset['terminals'][terminal_state] = 1
# # dataset['terminals'][-1] = 1
# observations = dataset['observations']
# actions = dataset['actions']
# rewards = dataset['rewards']
# next_observations = dataset['next_observations']
# terminals = dataset['terminals']

t = gym.make(env_name)
# dataset = t.get_dataset()
# if not dataset["terminals"][-1]:
#     dataset["timeouts"][-1] = True
# data = {}
# assert not np.any(dataset["terminals"] & dataset["timeouts"])
# data["states"] = dataset["observations"]
# data["actions"] = dataset["actions"]
# data["rewards"] = dataset["rewards"][:, None]
# data["done"] = (dataset["terminals"] | dataset["timeouts"])[:, None]
# data["is_finished"] = dataset["terminals"][:, None]
# assert data["done"][-1, 0]
# data["returns"] = np.zeros((data["states"].shape[0], 1))
# last = 0
# for i in range(data["returns"].shape[0] - 1, -1, -1):
#     last = data["rewards"][i, 0] + 0.99 * last * (1. - data["done"][i, 0])
#     data["returns"][i, 0] = last

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
trajectories = []
trajectory = []
s = 0

env_kwargs = t.env.env.spec.kwargs
def plot_cluster_cc(kwargs, points):
    maze_map = kwargs['maze_map']
    maze_size = kwargs['maze_size_scaling']

    start = None
    goals = []
    blocks = []
    # find start and goal positions
    for i in range(len(maze_map)):
        for j in range(len(maze_map[0])):
            if maze_map[i][j] == 'r':
                start = (len(maze_map) - i - 1, j)
            elif maze_map[i][j] == 'g':
                goals.append((len(maze_map) - i - 1, j))
            elif maze_map[i][j] == 1:
                blocks.append((len(maze_map) - i - 1, j))

    fig, ax = plt.subplots()
    # compute limit
    x_lim = (-(start[1] + 0.5) * maze_size, (len(maze_map[0]) - 0.5 - start[1]) * maze_size)
    y_lim = (-(start[0] + 0.5) * maze_size, (len(maze_map) - 0.5 - start[0]) * maze_size)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    # draw blocks
    for block in blocks:
        x, y = x_lim[0] + maze_size * block[1], y_lim[0] + maze_size * block[0]  # maze_size * block[0] - y_lim[1]
        ax.add_patch(Rectangle((x, y), maze_size, maze_size, linewidth=0, rasterized=True, color='#C0C0C0'))
    
    # draw clusters
    colors = ['#808080', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', 
            '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', 
            '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', 
            '#000075', '#ffffff', '#000000']
    # colors = [
    #         "#B84C9B", "#50F6E3", "#9397F9", "#8CED1C", "#777221", "#9BDE77", "#4103EC", "#E844F0",
    #         "#63A9BC", "#52E7A7", "#D5DF52", "#E785F9", "#71C407", "#F9EAC9", "#173581", "#CD11DD",
    #         "#22BA04", "#3BE745", "#BBCE06", "#38CBF1", "#48FF69", "#DA598D", "#24D34F", "#65208C",
    #         "#4769A5", "#57C547", "#75D515", "#855A8B", "#69C942", "#36143F", "#403E9B", "#AE1B43",
    #         "#98EF80", "#010992", "#0BF415", "#87AA1A", "#27D89F", "#58F9CC", "#CDBC70", "#950157",
    #         "#57C7BD", "#9C9B3E", "#72EF74", "#5E27B9", "#4BC7E7", "#9AE681", "#55C8B5", "#C89F93",
    #         "#F6D1B0", "#6C5F9D", "#7D085D", "#131C2D", "#418640", "#51591E", "#37E0D6", "#904389",
    #         "#BEB828", "#809716", "#52E4CF", "#0BD870", "#8347AB", "#2CB2C1", "#BE4BC1", "#2E8F71",
    #         "#F1059F", "#FF143B", "#A7738D", "#1A766E", "#4D46AF", "#2C8AC9", "#514335", "#5402C0",
    #         "#AB91B9", "#818C09", "#CA9DB1", "#998BB9", "#99C5EA", "#F225CC", "#1499BA", "#C3664C",
    #         "#6E1375", "#43E408", "#601FBC", "#52EBAB", "#7FFA0D", "#1EC789", "#37812F", "#D6C6B3",
    #         "#23D567", "#059FC6", "#FE6181", "#025770", "#503DA7", "#2B28B6", "#ABF6D3", "#36EC99",
    #         "#D4F8FD", "#32388D", "#3212E6", "#AF13D9"
    #         ]

    # for i in range(len(clusters)):
    #     points = []
    #     for j in range(len(clusters[i])):
    #         points.append(traj[clusters[i][j]])
    #     if len(points) != 0:
    #         points = np.concatenate(points)
    points[:, 1] = -points[:, 1]
    ax.scatter(points[:, 0], points[:, 1], s=0.01)

    # draw start and goal positions
    ax.plot(0, 0, 'ro')
    ax.annotate('start', (0, 0.25))
    for goal in goals:
        x = (goal[1] - start[1]) * maze_size
        y = (goal[0] - start[0]) * maze_size
        ax.plot(x, y, 'bo')
        ax.annotate('goal', (x, y + 0.25))
    plt.show()
    # # plt.savefig(f'dist_density/{name}.png')

    # fig.canvas.draw()  # Draw the canvas, cache the renderer
    # image_flat = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
    # # reversed converts (W, H) from get_width_height to (H, W)
    # image = image_flat.reshape(*reversed(fig.canvas.get_width_height()), 3)
    # plt.close()
    # return image

# Loop through the data and split into trajectories at terminal states
for obs, action, reward, terminal in zip(observations, actions, rewards, terminals):
    trajectory.append(obs[:2])
    if terminal:
        indices = int(np.random.randint(len(trajectory)-300, size=(1,)))
        plot_cluster_cc(env_kwargs, np.stack(trajectory)[indices:indices+300])
        # trajectories.append(trajectory)
        # for i in range(len(trajectory)):
        #     if i != 0:
        #         if not (trajectory[i-1][3] == trajectory[i][0]).all():
        #             print(s, i)
        #             assert 0
        print(s)
        s = 0
        trajectory = []
    s += 1

sampler = Traj_Data_Sampler(dataset)
episode_starts = sampler.sample_episode_starts(batch_size=20)
segments = sampler.sample(episode_starts, K=5)
print(len(segments), segments[0].shape)