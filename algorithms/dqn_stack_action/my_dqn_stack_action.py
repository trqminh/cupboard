import gym
from gym.spaces import Box, Discrete
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions.categorical import Categorical
from models import *
import time
import copy
import os
import numpy as np
import random

torch.manual_seed(1434)
np.random.seed(1434)


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = list()
        self.capacity = capacity
        self.cursor = 0

    def __len__(self):
        return len(self.memory)

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.cursor] = transition
        self.cursor = (self.cursor + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


def train(configs):
    env = gym.make(configs['env'])
    hidden_size = configs['hidden_size']
    lr = float(configs['lr'])
    n_episodes = configs['n_episodes']
    batch_size = configs['batch_size']
    device = configs['device']
    render = configs['render']
    gamma = configs['discount_factor']
    C = configs['reset_step']
    ep_thresh = configs['ep_thresh']

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    input_dim = obs_dim
    Q_fnc = mlp(input_dim, n_acts, hidden_size).to(device)
    Q_hat_fnc = mlp(input_dim, n_acts, hidden_size).to(device)
    Q_hat_fnc.load_state_dict(Q_fnc.state_dict())

    optimizer = optim.Adam(Q_fnc.parameters(), lr=lr)
    D = ReplayMemory(10000)

    Q_fnc.train()
    Q_hat_fnc.eval()
    step = 0

    for ep in range(n_episodes):
        loss = 1000000000.0
        episode_rew = 0.
        done = False
        obs = env.reset()

        while True:
            step += 1
            epsilon = random.uniform(0,1)
            if epsilon < ep_thresh:
                act = random.randint(0, n_acts - 1)
            else:
                obs = torch.from_numpy(obs).to(device=device, dtype=torch.float)
                act = torch.argmax(Q_fnc(obs)).item()

            pre_obs = copy.deepcopy(obs)
            obs, reward, done, info = env.step(act)
            episode_rew += reward
            D.push([pre_obs, act, reward, obs, done])

            if len(D) < batch_size:
                continue

            b_transitions = np.array(D.sample(batch_size))

            b_reward_mask = np.bitwise_xor(b_transitions[:,-1].astype(int), np.ones(batch_size, dtype=int))
            b_reward_mask = torch.from_numpy(b_reward_mask).to(device=device, dtype=torch.float)

            b_reward = np.stack(b_transitions[:,2])
            b_reward = torch.from_numpy(b_reward).to(device=device, dtype=torch.float)

            b_next_states = np.stack( b_transitions[:,3])
            b_next_states = torch.from_numpy(b_next_states).to(device=device, dtype=torch.float)

            y = b_reward + b_reward_mask * (Q_hat_fnc(b_next_states).max(1)[0]).detach().numpy() * gamma

            b_pre_states = np.stack(b_transitions[:,0])
            b_pre_states = torch.from_numpy(b_pre_states).to(device=device, dtype=torch.float)

            b_acts = np.stack(b_transitions[:,1])
            b_acts = torch.from_numpy(b_acts).to(device=device, dtype=torch.long).unsqueeze(1)

            pred_Q = torch.gather(Q_fnc(b_pre_states), 1, b_acts)

            optimizer.zero_grad()
            loss = torch.mean((y - pred_Q)**2)
            loss.backward()
            optimizer.step()

            if step >= C:
                Q_hat_fnc.load_state_dict(Q_fnc.state_dict())
                step = 0

            if done:
                break

        if ep % 10 == 0:
            print('Episode {}, loss: {:.2f}, episode_reward: {:.2f}'.format(ep, loss, episode_rew))

