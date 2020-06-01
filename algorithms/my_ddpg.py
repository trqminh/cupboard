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
        self.n_episodes = configs['n_episodes']
        self.memory_size = configs['memory_size']
        self.gamma = configs['gamma']
        self.target_update_step = configs['target_update_step']
        self.global_step = 0

        # save the "best" policy
        self.best_mean_episode_ret = -1e6
        self.best_policy_state_dict = None
        self.trained_model_path = configs['trained_model_path']

        # set up actor-critic (or policy - q_fucntion)
        self.obs_dim = self.env.observation_space.shape[0]
        self.n_acts = self.env.action_space.shape[0]
        output_activation = nn.Tanh

        self.actor = mlp(sizes = [self.obs_dim] + self.hidden_sizes + [self.n_acts], 
                        output_activation=output_activation).to(self.device)
        self.target_actor = mlp(sizes = [self.obs_dim] + self.hidden_sizes + [self.n_acts], 
                        output_activation=output_activation).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.critic = Critic(self.obs_dim, self.n_acts)
        self.target_critic = Critic(self.obs_dim, self.n_acts)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.replay_memory = self.ReplayMemory(self.memory_size)
        self.optimizer = optim.Adam(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                lr=self.lr
            )

    def optimize(self):
        # SAMPLE FROM REPLAY MEMORY
        batch = self.Transition(*zip(*transitions))

        batch_done_mask = torch.cat(batch.done_mask)
        batch_reward = torch.cat(batch.reward)
        batch_state = torch.cat(batch.state)
        batch_act = torch.cat(batch.act)
        batch_next_state = torch.cat(batch.next_state)

        target = batch_reward + self.gamma * batch_done_mask * \
                (self.target_critic(batch_next_state, self.target_actor(batch_next_state)).detach())
        predict_target = self.critic(batch_state, self.actor(batch_state))
        loss = torch.mean((target - predict_target)**2)

        # OPTIMIZE ACTOR AND CRITIC
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # RESET TARGET NETWORK
        if self.global_step > 0 and self.global_step % self.target_update_step == 0:
            self.target_actor.load_state_dict(self.actor.state_dict())
            self.target_critic.load_state_dict(self.critic.state_dict())
            # TODO: how to update as ddpg type



    def train(self):
        # TRAINING
        for ep in range(self.n_episodes):
            ep_ret, done = 0., False
            obs = torch.from_numpy(self.env.reset()).to(device=device, dtype=torch.float).unsqueeze(0)

            for t in count():
                self.global_step += 1
                # SELECT ACTION
                with torch.no_grad():
                    act = self.actor(obs) # TODO: + epsilone ~ normal(0,1)

                # EXCUTE ACTION AND STORE TRANSITION
                next_obs, reward, done, _ = env.step(act.tolist())
                ep_ret += reward

                next_obs = torch.from_numpy(next_obs).to(device=device,dtype=torch.float).unsqueeze(0)
                reward = torch.tensor([float(reward)], device=device)
                done_mask = torch.tensor([1. - float(done)], device=device)

                self.replay_memory.push([obs, act, reward, next_obs, done_mask])
                obs = next_obs

                # OPTIMIZATION
                if len(D) >= batch_size: # TODO: and time to update
                    self.optimize()

                if done:
                    ep_rets.append(ep_ret)
                    ep_lens.append(t + 1)
                    break

            # TODO: what is need to log
            if ep % 20 == 0:
                print('Episode {}, mean episode length: {:.2f}, mean episode return: {:.2f}'.format(
                    ep, np.mean(ep_lens), np.mean(ep_rets)))
                ep_rets = []
                ep_lens = []

