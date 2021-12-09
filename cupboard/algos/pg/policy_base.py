import gym
from gym.spaces import Box, Discrete
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from .mlp import MLPPolicy
import time
import copy
import os
import numpy as np


class PolicyBase(object):
    def __init__(self, env_fn, 
                    batch_size,
                    device,
                    render,
                    lr,
                    n_epochs,
                    trained_model_path,
                    mlp_hidden_sizes=[32, 32],
                    policy=MLPPolicy):

        self.env = env_fn
        assert isinstance(self.env.observation_space, Box), \
            "This example only works for envs with continuous state spaces."
        self.batch_size = batch_size
        self.device = device
        self.render = render
        self.lr = lr
        self.n_epochs = n_epochs
        self.log_std = None # for continuous action spaces
        self.is_continuous = isinstance(self.env.action_space, Box)

        # save the "best" policy
        self.best_mean_episode_ret = -1e6
        self.best_policy_state_dict = None
        self.trained_model_path = trained_model_path

        # set up policy
        self.obs_dim = self.env.observation_space.shape[0]
        self.n_acts = None

        if self.is_continuous:
            self.n_acts = self.env.action_space.shape[0]
            self.log_std = torch.tensor(-0.5*np.ones(self.n_acts), 
                    dtype=torch.float32, requires_grad=True, device=self.device)
        else:
            self.n_acts = self.env.action_space.n

        self.policy = policy(obs_dim=self.obs_dim, act_dim=self.n_acts, layer_sizes=mlp_hidden_sizes).to(self.device)
        print('Policy architecture: ', self.policy)

    @staticmethod
    def reward_to_go(rews):
        n = len(rews)
        rtgs = np.zeros_like(rews)
        for i in reversed(range(n)):
            rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)

        return rtgs

    def train(self):
        batch_obs, batch_acts, batch_weights, batch_rets, batch_lens = \
                [], [], [], [], []
        obs, done, ep_rews = self.env.reset(), False, []

        while True:
            if self.render:
                self.env.render()

            batch_obs.append(obs.copy()) # save the observation for offline update
            logit = self.policy(torch.from_numpy(obs).to(dtype=torch.float, device=self.device))

            if self.is_continuous:
                logit = nn.Tanh()(logit)
                std = torch.exp(self.log_std)
                distribution = Normal(logit, std) # logit as mean
                act = distribution.sample()
                '''
                for i in range(self.n_acts):
                    act[i] = torch.clamp(act[i], self.env.action_space.low[i], 
                            self.env.action_space.high[i])

                print(act)
                '''
            else:
                distribution = Categorical(F.softmax(logit, dim=0))
                act = distribution.sample()

            obs, reward, done, info = self.env.step(act.tolist())
            batch_acts.append(act.tolist())
            ep_rews.append(reward)

            if done:
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                # batch_rets and batch_lens just for information
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                batch_weights += list(self.reward_to_go(ep_rews)) # batch_weights += ([ep_ret]*ep_len)
                obs, done, ep_rews = self.env.reset(), False, []

                if len(batch_obs) > self.batch_size:
                    break

        epoch_info = {
                'n_episodes': len(batch_rets),
                'mean_epi_return': np.mean(batch_rets),
                'mean_epi_len': np.mean(batch_lens)
                }

        if epoch_info['mean_epi_len'] > self.best_mean_episode_ret:
            self.best_mean_episode_ret = epoch_info['mean_epi_return']
            self.best_policy_state_dict = copy.deepcopy(self.policy.state_dict())

        batch_obs = torch.tensor(batch_obs).to(self.device)
        batch_acts = torch.tensor(batch_acts).to(self.device)
        batch_weights = torch.tensor(batch_weights).to(self.device)

        return batch_obs, batch_acts, batch_weights, epoch_info

    def perform(self):
        # In this perform function, I skip the learned log_std, just perform by its mean.
        if os.path.exists(self.trained_model_path):
            self.policy.load_state_dict(torch.load(self.trained_model_path, map_location=self.device))
        else:
            print('Trained model does not exist, performed with random initialization')
        obs, done, ep_rews = self.env.reset(), False, []

        while True:
            self.env.render()
            logit = self.policy(torch.from_numpy(obs).to(dtype=torch.float, device=self.device))

            if self.is_continuous:
                std = torch.exp(self.log_std)
                distribution = Normal(logit, std) # logit as mean
                act = distribution.sample()
            else:
                distribution = Categorical(F.softmax(logit, dim=0))
                act = distribution.sample()

            obs, reward, done, info = self.env.step(act.tolist())
            ep_rews.append(reward)

            if done:
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                print('Episode return: {:.2f}, episode len: {:.2f}'.format(ep_ret, ep_len))
                obs, done, ep_rews = self.env.reset(), False, [] 

