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


class DQN(object):
    def __init__(self, configs, env):
        # GET CONFIG
        self.env = env
        self.hidden_sizes = configs['hidden_sizes']
        self.lr = float(configs['lr'])
        self.n_episodes = configs['n_episodes']
        self.batch_size = configs['batch_size']
        self.device = configs['device']
        self.render = configs['render']
        self.gamma = configs['discount_factor']
        self.target_update_step = configs['target_update_step']
        self.epsilon_decay = configs['epsilon_decay']
        self.replay_mem_size = configs['replay_mem_size']
        self.learning_starts = configs['learning_starts']
        self.learning_freq = configs['learning_freq']
        self.epsilon = 1.

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


    def train(self):
        # NETWORK DECLARATION
        obs_dim = self.env.observation_space.shape[0]
        n_acts = self.env.action_space.n
        input_dim = obs_dim

        Q_net = mlp([input_dim] + self.hidden_sizes + [n_acts]).to(self.device)
        Q_target_net = mlp([input_dim] + self.hidden_sizes + [n_acts]).to(self.device)
        Q_target_net.load_state_dict(Q_net.state_dict())

        optimizer = optim.Adam(Q_net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        replay_mem = self.ReplayMemory(self.replay_mem_size)
        Q_net.train()
        Q_target_net.eval()
        ep_rets = []
        ep_lens = []
        global_step = 0
        debug = False
        intense, prosaic = 0, 0

        # TRAINING
        for ep in range(self.n_episodes + 1):
            loss, ep_ret, done = None, 0., False
            obs = torch.from_numpy(self.env.reset()).to(device=self.device, 
                    dtype=torch.float).unsqueeze(0)

            for t in count():
                global_step += 1
                # SELECT ACTION WITH EPSILON GREEDY
                if self.epsilon < random.uniform(0,1):
                    prosaic += 1
                    with torch.no_grad():
                        act = Q_net(obs).max(1)[1].view(1,1)
                else:
                    intense += 1
                    act = torch.tensor([[self.env.action_space.sample()]], 
                            device=self.device, dtype=torch.long)

                # EXCUTE ACTION AND STORE TRANSITION
                next_obs, reward, done, _ = self.env.step(act.item())
                ep_ret += reward

                next_obs = torch.from_numpy(next_obs).to(device=self.device,
                        dtype=torch.float).unsqueeze(0)
                reward = torch.tensor([float(reward)], device=self.device)
                done_mask = torch.tensor([1. - float(done)], device=self.device)

                replay_mem.push([obs, act, reward, next_obs, done_mask])
                obs = next_obs

                # OPTIMIZATION
                #if len(replay_mem) < self.batch_size or \
                #        global_step < self.learning_starts:
                #    continue
                if len(replay_mem) < self.batch_size:
                    continue

                if not debug:
                    print('global_step ', global_step)
                    print('episode ', ep)
                    debug = True

                transitions = replay_mem.sample(self.batch_size)
                batch = Transition(*zip(*transitions))

                batch_done_mask = torch.cat(batch.done_mask)
                batch_reward = torch.cat(batch.reward)
                batch_state = torch.cat(batch.state)
                batch_act = torch.cat(batch.act)
                batch_next_state = torch.cat(batch.next_state)

                y = batch_reward + batch_done_mask * self.gamma * \
                        (Q_target_net(batch_next_state).max(1)[0].detach())
                predict_Q = torch.gather(Q_net(batch_state), 1, batch_act)

                loss = criterion(predict_Q, y.unsqueeze(1))
                #print(loss)
                loss = torch.clamp(loss, -1, 1)
                optimizer.zero_grad()
                loss.backward()
                for param in Q_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()

                # RESET TARGET Q NETWORK
                if global_step % self.target_update_step == 0:
                    Q_target_net.load_state_dict(Q_net.state_dict())

                if done:
                    self.epsilon = max(self.epsilon * self.epsilon_decay, 0.)
                    ep_rets.append(ep_ret)
                    ep_lens.append(t + 1)
                    break

            if ep % 100 == 0:
                print(len(ep_rets))
                print(global_step)
                print(len(ep_lens))
                print('Episode {}, loss: {:.2f}, mean episode length: {:.2f}, current 100 ep mean episode return: {:.2f}'\
                        .format(ep, loss, np.mean(ep_lens), np.mean(ep_rets)))
                ep_rets = []
                ep_lens = []

        torch.save(Q_net.state_dict(), self.trained_model_path)

    def perform(self):
        print("I am performing!!")
        self.env.reset()
        while True:
            self.env.render(self.render)
            act = self.env.action_space.sample()
            self.env.step(act)
            print(act)

