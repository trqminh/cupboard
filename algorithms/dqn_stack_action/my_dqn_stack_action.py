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
from ast import literal_eval

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
    # GET CONFIG
    env = gym.make(configs['env'])
    hidden_sizes = literal_eval(configs['hidden_sizes'])
    lr = float(configs['lr'])
    n_episodes = configs['n_episodes']
    batch_size = configs['batch_size']
    device = configs['device']
    render = configs['render']
    gamma = configs['discount_factor']
    C = configs['reset_step']
    ep_thresh = configs['ep_thresh']
    replay_mem_size = configs['replay_mem_size']

    # NETWORK DECLARATION
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n
    input_dim = obs_dim

    Q_net = mlp([input_dim] + hidden_sizes + [n_acts]).to(device)
    Q_target_net = mlp([input_dim] + hidden_sizes + [n_acts]).to(device)
    Q_target_net.load_state_dict(Q_net.state_dict())
    optimizer = optim.Adam(Q_net.parameters(), lr=lr)

    D = ReplayMemory(replay_mem_size)
    Q_net.train()
    Q_target_net.eval()
    step = 0
    ep_rets = []

    # TRAINING
    for ep in range(n_episodes):
        loss = 1000000000.0
        ep_ret = 0.
        done = False
        obs = env.reset()

        while True:
            # SELECT ACTION WITH EPSILON GREEDY
            epsilon = random.uniform(0,1)
            if epsilon < ep_thresh:
                act = random.randint(0, n_acts - 1)
            else:
                obs_t = torch.from_numpy(obs).to(device=device, dtype=torch.float)
                act = torch.argmax(Q_net(obs_t)).item()

            # EXCUTE ACTION AND STORE TRANSITION
            pre_obs = copy.deepcopy(obs)
            obs, reward, done, info = env.step(act)
            ep_ret += reward
            D.push([pre_obs, act, reward, obs, done])

            # SAMPLE MINIBATCH AND OPTIMIZE
            if len(D) < batch_size:
                continue

            # compute target value
            b_transitions = np.array(D.sample(batch_size))

            b_reward_mask = np.bitwise_xor(b_transitions[:,-1].astype(int), np.ones(batch_size, dtype=int))
            b_reward_mask = torch.from_numpy(b_reward_mask).to(device=device, dtype=torch.float)

            b_reward = np.stack(b_transitions[:,2])
            b_reward = torch.from_numpy(b_reward).to(device=device, dtype=torch.float)

            b_next_states = np.stack(b_transitions[:,3])
            b_next_states = torch.from_numpy(b_next_states).to(device=device, dtype=torch.float)

            y = b_reward + b_reward_mask * (Q_target_net(b_next_states).max(1)[0]) * gamma

            # compute predict value
            b_pre_states = np.stack(b_transitions[:,0])
            b_pre_states = torch.from_numpy(b_pre_states).to(device=device, dtype=torch.float)

            b_acts = np.stack(b_transitions[:,1])
            b_acts = torch.from_numpy(b_acts).to(device=device, dtype=torch.long).unsqueeze(1)

            pred_Q = torch.gather(Q_net(b_pre_states), 1, b_acts)

            # optimize mse
            optimizer.zero_grad()
            loss = torch.mean((y - pred_Q)**2)
            loss.backward()
            optimizer.step()

            # RESET TARGET Q NETWORK
            step += 1
            if step >= C:
                Q_target_net.load_state_dict(Q_net.state_dict())
                step = 0

            if done:
                ep_rets.append(ep_ret)
                break

        if ep % 50 == 0:
            print('Episode {}, loss: {:.2f}, mean episode return: {:.2f}'.format(ep, loss, np.mean(ep_rets)))
            ep_rets = []

