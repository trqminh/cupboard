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


def one_hot_vec(act, n_acts):
    ohv = [0.] * n_acts
    ohv[act] = 1.
    return ohv


def argmax_and_max(Q_fnc, obs, n_acts):
    device = next(Q_fnc.parameters()).device
    Q_max, act_max = 0, 0
    for act in range(n_acts):
        Q_input = torch.from_numpy(np.append(obs, one_hot_vec(act, n_acts))).to(device=device, dtype=torch.float)
        Q_value = Q_fnc(Q_input).item()
        if Q_value > Q_max:
            act_max = act
            Q_max = Q_value

    return act_max, Q_max


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

    input_dim = obs_dim + n_acts # plus one hot action
    Q_fnc = mlp(input_dim, 1, hidden_size).to(device)
    Q_hat_fnc = mlp(input_dim, 1, hidden_size).to(device)
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
                with torch.no_grad():
                    act, _ = argmax_and_max(Q_fnc, obs, n_acts)

            pre_obs = copy.deepcopy(obs)
            obs, reward, done, info = env.step(act)
            episode_rew += reward
            D.push([pre_obs, act, reward, obs, done])

            if len(D) < batch_size:
                continue

            optimizer.zero_grad()
            b_transitions = D.sample(batch_size)
            y = np.zeros([batch_size, 1])

            for i, transition in enumerate(b_transitions):
                if transition[-1] == True:
                    y[i] = b_transitions[i][2]
                else:
                    with torch.no_grad():
                        act_max, Q_value_max = argmax_and_max(Q_hat_fnc, b_transitions[i][0], n_acts)
                        y[i] = b_transitions[i][2] + gamma*Q_value_max

            Q_input = np.zeros([batch_size, obs_dim + n_acts])
            for i in range(batch_size):
                Q_input[i] = np.append(b_transitions[i][0], one_hot_vec(b_transitions[i][1], n_acts))

            y = torch.from_numpy(y).to(device=device, dtype=torch.float)
            Q_input = torch.from_numpy(Q_input).to(device=device, dtype=torch.float)

            loss = torch.mean((y - Q_fnc(Q_input))**2)
            loss.backward()
            optimizer.step()

            if step >= C:
                Q_hat_fnc.load_state_dict(Q_fnc.state_dict())
                step = 0

            if done:
                break

        if ep % 10 == 0:
            print('Episode {}, loss: {:.2f}, episode_reward: {:.2f}'.format(ep, loss, episode_rew))
