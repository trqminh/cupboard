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

        self.critic = Critic(self.obs_dim, self.n_acts)
        self.target_critic = Critic(self.obs_dim, self.n_acts)
        # TODO: get the target copy the original
        self.replay_memory = self.ReplayMemory(self.memory_size)

    def train(self):
        # TRAINING
        for ep in range(self.n_episodes):
            loss = None
            ep_ret = 0.
            done = False
            obs = torch.from_numpy(self.env.reset()).to(device=device, dtype=torch.float).unsqueeze(0)

            for t in count():
                global_step += 1
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
                if len(D) < batch_size: # TODO: and time to update
                    continue

                # TODO: split optimize into a method
                transitions = self.replay_memory.sample(batch_size)
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

                # TODO: optimizer and target update
                '''
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
                '''

            # TODO: what is need to log
            if ep % 20 == 0:
                print('Episode {}, loss: {:.2f}, mean episode length: {:.2f}, mean episode return: {:.2f}'.format(
                    ep, loss, np.mean(ep_lens), np.mean(ep_rets)))
                ep_rets = []
                ep_lens = []

