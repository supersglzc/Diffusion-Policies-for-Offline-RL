import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_


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
    
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 10)
        self.output = nn.Linear(10, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.output(x)
        return x

def compute_reward(samples, center, max_reward, max_distance=1.0):
    distances = np.sqrt((samples[:, 0] - center[0]) ** 2 + (samples[:, 1] - center[1]) ** 2)
    rewards = max_reward * (1 - distances / max_distance)
    rewards = np.maximum(0, rewards)
    
    return rewards