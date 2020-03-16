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


def argmax_and_max(Q_fnc, obs, n_acts):
    device = next(Q_fnc.parameters()).device
    Q_max, act_max = 0, 0
    for act in range(n_acts):
        Q_input = torch.from_numpy(np.append(obs, act)).to(device=device, dtype=torch.float)
        Q_value = Q_fnc(Q_input).item()
        if Q_value > Q_max:
            act_max = act
            Q_max = Q_value

    return act_max, Q_max


def train(configs):
    env = gym.make(configs['env'])
    hidden_size = configs['hidden_size']
    lr = float(configs['lr'])
    n_epochs = configs['n_epochs']
    batch_size = configs['batch_size']
    device = configs['device']
    render = configs['render']
    gamma = configs['discount_factor']
    C = configs['reset_step']

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    input_dim = obs_dim + 1 # plus action value
    Q_fnc = mlp(input_dim, 1, hidden_size).to(device)
    Q_hat_fnc = copy.deepcopy(Q_fnc)
    optimizer = optim.Adam(Q_fnc.parameters(), lr=lr)


    done = False
    obs = env.reset()
    D = ReplayMemory(10000)
    step = 0

    for ep in range(n_epochs):
        loss = 1000000000.0
        episode_rew = 0.
        while True:
            step += 1
            epsilon = random.uniform(0,1)
            if epsilon < 0.15:
                act = random.randint(0, n_acts - 1)
            else:
                act, _ = argmax_and_max(Q_fnc, obs, n_acts)

            pre_obs = copy.deepcopy(obs)
            obs, reward, done, info = env.step(act)
            episode_rew += reward
            D.push([pre_obs, act, reward, obs, done])

            if done:
                obs, done = env.reset(), False
                break

            if len(D) < batch_size:
                continue

            b_transitions = D.sample(batch_size)
            y = np.zeros([batch_size, 1])

            for i, transition in enumerate(b_transitions):
                if transition[-1] == True:
                    y[i] = b_transitions[i][2]
                else:
                    act_max, Q_value_max = argmax_and_max(Q_hat_fnc, b_transitions[i][0], n_acts)
                    y[i] = b_transitions[i][2] + gamma*Q_value_max

            y = torch.from_numpy(y).to(device=device, dtype=torch.float)

            Q_input = np.zeros([batch_size, obs_dim + 1])
            for i in range(batch_size):
                Q_input[i] = np.append(b_transitions[i][0], b_transitions[i][1])

            Q_input = torch.from_numpy(Q_input).to(device=device, dtype=torch.float)

            loss = torch.mean((y - Q_fnc(Q_input))**2)
            loss.backward()
            optimizer.step()

            if step >= C:
                Q_hat_fnc = copy.deepcopy(Q_fnc)
                step = 0

        if ep % 10 == 0:
            print('Episode {}, loss: {:.2f}, episode_reward: {:.2f}'.format(ep, loss, episode_rew))

