import gym
from gym.spaces import Box, Discrete
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from networks import *
import time
import copy
import os
import numpy as np
from itertools import count
from collections import namedtuple
import random


class DDPG(object):
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

    def __init__(self, configs, env):
        self.env = env
        assert isinstance(self.env.observation_space, Box), \
            "This example only works for envs with continuous state spaces."
        assert isinstance(self.env.action_space, Box), \
            "This example only works for envs with continuous action spaces."

        self.hidden_sizes = configs['hidden_sizes']
        self.batch_size = configs['batch_size']
        self.device = configs['device']
        self.render = configs['render']
        self.lr = float(configs['lr'])
        self.memory_size = configs['memory_size']
        self.gamma = configs['discount_factor']
        self.update_after = configs['update_after']
        self.update_every = configs['update_every']
        self.max_episode_len = configs['max_episode_len']
        self.steps_per_epoch = configs['steps_per_epoch']
        self.n_epochs = configs['n_epochs']
        self.steps_per_epoch = configs['steps_per_epoch']
        self.polyak = configs['polyak']

        # save the "best" policy
        self.best_mean_episode_ret = -1e6
        self.best_policy_state_dict = None
        self.trained_model_path = configs['trained_model_path']

        # set up actor-critic (or policy - q_function)
        self.obs_dim = self.env.observation_space.shape[0]
        self.n_acts = self.env.action_space.shape[0]
        output_activation = nn.Tanh

        self.actor = mlp(sizes = [self.obs_dim] + self.hidden_sizes + [self.n_acts], 
                        output_activation=output_activation).to(self.device)
        self.target_actor = mlp(sizes = [self.obs_dim] + self.hidden_sizes + [self.n_acts], 
                        output_activation=output_activation).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.critic = Critic(self.obs_dim, self.n_acts).to(self.device)
        self.target_critic = Critic(self.obs_dim, self.n_acts).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.replay_memory = self.ReplayMemory(self.memory_size)
        self.optimizer = optim.Adam(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                lr=self.lr
            )

    def optimize(self):
        # SAMPLE FROM REPLAY MEMORY
        transitions = self.replay_memory.sample(self.batch_size)
        batch = self.Transition(*zip(*transitions))

        batch_done_mask = torch.cat(batch.done_mask)
        batch_reward = torch.cat(batch.reward)
        batch_state = torch.cat(batch.state)
        batch_act = torch.cat(batch.act)
        batch_next_state = torch.cat(batch.next_state)

        target = batch_reward + self.gamma * batch_done_mask * \
                (self.target_critic(batch_next_state, self.target_actor(batch_next_state)).detach())
        predict_target = self.critic(batch_state, batch_act)
        actor_loss = torch.mean((target - predict_target)**2)
        critic_loss = -torch.mean(self.critic(batch_state, self.actor(batch_state)))
        loss = actor_loss + critic_loss

        # OPTIMIZE ACTOR AND CRITIC
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # UPDATE TARGET NETWORK
        with torch.no_grad():
            for actor_p, target_actor_p, critic_p, target_critic_p in zip(
                    self.actor.parameters(), 
                    self.target_actor.parameters(),
                    self.critic.parameters(),
                    self.target_critic.parameters()
                    ):
                target_actor_p.data.mul_(self.polyak)
                target_actor_p.data.add_((1 - self.polyak) * actor_p.data)
                target_critic_p.data.mul_(self.polyak)
                target_critic_p.data.add_((1 - self.polyak) * critic_p.data)


    def train(self):
        # TRAINING
        ep_rets = []
        ep_lens = []
        ep_ret, ep_len, done = 0., 0, False
        obs = torch.from_numpy(self.env.reset()).to(device=self.device, dtype=torch.float).unsqueeze(0)
        for global_step in range(self.n_epochs * self.steps_per_epoch):
            # SELECT ACTION
            with torch.no_grad():
                act = self.actor(obs).squeeze(0)
                act_noise = torch.randn_like(act)
                act = act + act_noise
                low = torch.tensor(self.env.action_space.low).to(self.device)
                high = torch.tensor(self.env.action_space.high).to(self.device)
                act = torch.max(torch.min(act, high), low)

            # EXECUTE ACTION AND STORE TRANSITION
            next_obs, reward, done, _ = self.env.step(act.tolist())
            ep_ret += reward
            ep_len += 1

            next_obs = torch.from_numpy(next_obs).to(device=self.device,dtype=torch.float).unsqueeze(0)
            reward = torch.tensor([float(reward)], device=self.device)
            done_mask = torch.tensor([1. - float(done)], device=self.device)

            self.replay_memory.push([obs, act.unsqueeze(0), reward, next_obs, done_mask])
            obs = next_obs

            if done or ep_len >= self.max_episode_len:
                ep_rets.append(ep_ret)
                ep_lens.append(ep_len)
                ep_ret, ep_len, done = 0., 0, False
                obs = torch.from_numpy(self.env.reset()).to(device=self.device, 
                        dtype=torch.float).unsqueeze(0)

            # OPTIMIZATION
            if global_step > self.update_after and global_step % self.update_every == 0:
                for _ in range(self.update_every):
                    self.optimize()

            # LOGGING
            if global_step % (self.steps_per_epoch * 20) == 0 and global_step > 0:
                print('Epoch {}, mean episode length: {:.2f}, mean episode return: {:.2f}'.format(
                    global_step//self.steps_per_epoch, np.mean(ep_lens), np.mean(ep_rets)))
                ep_rets = []
                ep_lens = []

