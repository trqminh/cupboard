import gym
from gym.spaces import Box, Discrete
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions.categorical import Categorical
from networks import *
import time
import copy
import os
import numpy as np
import random
from ast import literal_eval
from itertools import count
from collections import namedtuple

torch.manual_seed(1434)
np.random.seed(1434)

Transition = namedtuple('Transition',
                        ('state', 'act', 'reward', 'next_state', 'done_mask'))

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
    hidden_sizes = configs['hidden_sizes']
    lr = float(configs['lr'])
    n_episodes = configs['n_episodes']
    batch_size = configs['batch_size']
    device = configs['device']
    render = configs['render']
    gamma = configs['discount_factor']
    target_update_step = configs['target_update_step']
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
    criterion = nn.MSELoss()

    D = ReplayMemory(replay_mem_size)
    Q_net.train()
    Q_target_net.eval()
    ep_rets = []
    ep_lens = []
    global_step = 0

    # TRAINING
    for ep in range(n_episodes):
        loss = None
        ep_ret = 0.
        done = False
        obs = torch.from_numpy(env.reset()).to(device=device, dtype=torch.float).unsqueeze(0)

        for t in count():
            global_step += 1
            # SELECT ACTION WITH EPSILON GREEDY
            epsilon = random.uniform(0,1)
            if epsilon < ep_thresh:
                act = torch.tensor([[random.randint(0, n_acts - 1)]], device=device, dtype=torch.long)
            else:
                with torch.no_grad():
                    act = Q_net(obs).max(1)[1].view(1,1)

            # EXCUTE ACTION AND STORE TRANSITION
            next_obs, reward, done, _ = env.step(act.item())
            ep_ret += reward

            next_obs = torch.from_numpy(next_obs).to(device=device,dtype=torch.float).unsqueeze(0)
            reward = torch.tensor([float(reward)], device=device)
            done_mask = torch.tensor([1. - float(done)], device=device)

            D.push([obs, act, reward, next_obs, done_mask])
            obs = next_obs

            # OPTIMIZATION
            if len(D) < batch_size:
                continue

            transitions = D.sample(batch_size)
            batch = Transition(*zip(*transitions))

            batch_done_mask = torch.cat(batch.done_mask)
            batch_reward = torch.cat(batch.reward)
            batch_state = torch.cat(batch.state)
            batch_act = torch.cat(batch.act)
            batch_next_state = torch.cat(batch.next_state)

            y = batch_reward + batch_done_mask * gamma * (Q_target_net(batch_next_state).max(1)[0].detach())
            predict_Q = torch.gather(Q_net(batch_state), 1, batch_act)

            loss = criterion(predict_Q, y.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            for param in Q_net.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()

            # RESET TARGET Q NETWORK
            if global_step % target_update_step == 0:
                Q_target_net.load_state_dict(Q_net.state_dict())

            if done:
                ep_rets.append(ep_ret)
                ep_lens.append(t + 1)
                break

        if ep % 20 == 0:
            print('Episode {}, loss: {:.2f}, mean episode length: {:.2f}, mean episode return: {:.2f}'.format(
                ep, loss, np.mean(ep_lens), np.mean(ep_rets)))
            ep_rets = []
            ep_lens = []

