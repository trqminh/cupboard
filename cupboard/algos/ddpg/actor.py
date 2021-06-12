# Taken from spinnup ddpg
import torch
import torch.nn as nn
from .mlp import mlp


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits (of paramount importance).
        return self.act_limit * self.pi(obs)

