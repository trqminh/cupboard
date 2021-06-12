# taken from spinning up openai
import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, layer_sizes=[128, 128]):
        super(MLPPolicy, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.layer_sizes = layer_sizes


    def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
        # Build a feedforward neural network.
        layers = []
        for j in range(len(sizes)-1):
            act = activation if j < len(sizes)-2 else output_activation
            layers += [nn.Linear(sizes[j], sizes[j+1]), act()]

        return nn.Sequential(*layers)

    def forwad(self, x):
        return mlp(sizes=[obs_dim] + self.layer_sizes + [act_dim])(x)

