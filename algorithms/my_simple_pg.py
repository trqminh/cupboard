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
from .policy_base import PolicyBase


class SimplePolicyGradient(PolicyBase):
    def __init__(self, configs, env):
        super().__init__(configs, env)

    def train(self):
        # optimizer and things
        params = list(self.policy.parameters())
        if self.is_continuous:
            params.append(self.log_std)
        optimizer = optim.Adam(params, lr=self.lr)

        for epoch in range(self.n_epochs):
            self.policy.train()
            batch_obs, batch_acts, batch_weights, epoch_info = super().train()

            batch_logits = self.policy(batch_obs)
            batch_log_prob = None

            if self.is_continuous:
                batch_log_std = torch.cat([self.log_std.unsqueeze(0)] * batch_logits.shape[0])
                batch_std = torch.exp(batch_log_std)
                batch_distribution = Normal(batch_logits, batch_std)
                batch_log_prob = batch_distribution.log_prob(batch_acts).sum(axis=-1)
            else:
                batch_distribution = Categorical(F.softmax(batch_logits, dim=1))
                batch_log_prob = batch_distribution.log_prob(batch_acts)

            loss = torch.mean(-batch_log_prob * batch_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print('Epoch {}, loss: {}, epoch info: {}'.format(epoch, loss,\
                        epoch_info))

        torch.save(self.best_policy_state_dict, self.trained_model_path)

    def perform(self):
        super().perform()
