import time
import math
import torch
import numpy as np


class Traj_Data_Sampler(object):
	def __init__(self, data, device='cuda', reward_tune='no'):
		self.state = torch.from_numpy(data['observations']).float().to(device)
		self.action = torch.from_numpy(data['actions']).float().to(device)
		self.next_state = torch.from_numpy(data['next_observations']).float().to(device)
		reward = torch.from_numpy(data['rewards']).view(-1, 1).float().to(device)
		self.not_done = 1. - torch.from_numpy(data['terminals']).view(-1, 1).float().to(device)

		self.size = self.state.shape[0]
		self.state_dim = self.state.shape[1]
		self.action_dim = self.action.shape[1]

		self.device = device

		if reward_tune == 'normalize':
			reward = (reward - reward.mean()) / reward.std()
		elif reward_tune == 'iql_antmaze':
			reward = reward - 1.0
		elif reward_tune == 'iql_locomotion':
			reward = iql_normalize(reward, self.not_done)
		elif reward_tune == 'cql_antmaze':
			reward = (reward - 0.5) * 4.0
		elif reward_tune == 'antmaze':
			reward = (reward - 0.25) * 2.0
		self.reward = reward
		self.episode_starts = torch.tensor([0] + (np.where(data['terminals'][:-1])[0] + 1).tolist()).to(device)
		self.min_length = self.get_minimum_traj_length(data['terminals'])

	def sample(self, episode_start, K, sample='start'):
		if K > self.min_length:
			raise RuntimeError(
                f"segment length {K} is greater than the trajectory length {self.min_length}"
            )
		
		if sample == 'start':
			offset = torch.zeros_like(episode_start, device=episode_start.device)
		elif sample == 'random':
			offset = torch.randint(0, self.min_length - K - 1, size=(episode_start.shape[0],))
		else:
			raise NotImplementedError

		return (
			self.state[episode_start + offset:episode_start + offset + K].to(self.device),
			self.action[episode_start + offset:episode_start + offset + K].to(self.device),
			self.next_state[episode_start + offset:episode_start + offset + K].to(self.device),
			self.reward[episode_start + offset:episode_start + offset + K].to(self.device),
			self.not_done[episode_start + offset:episode_start + offset + K].to(self.device)
        )
	
	def sample_episode_starts(self, batch_size):
		ind = torch.randint(0, self.episode_starts.shape[0], size=(batch_size,))
		return self.episode_starts[ind]
		
	def get_minimum_traj_length(self, terminals):
		end_indices = np.where(terminals)[0]
		episode_lengths = np.diff(np.concatenate(([0], end_indices + 1)))
		min_length = np.min(episode_lengths)
		return min_length


def iql_normalize(reward, not_done):
	trajs_rt = []
	episode_return = 0.0
	for i in range(len(reward)):
		episode_return += reward[i]
		if not not_done[i]:
			trajs_rt.append(episode_return)
			episode_return = 0.0
	rt_max, rt_min = torch.max(torch.tensor(trajs_rt)), torch.min(torch.tensor(trajs_rt))
	reward /= (rt_max - rt_min)
	reward *= 1000.
	return reward
