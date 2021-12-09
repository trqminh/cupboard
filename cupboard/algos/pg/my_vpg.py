import gym
from gym.spaces import Box, Discrete
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from .policy_base import PolicyBase
from .mlp import MLPPolicy


class VanillaPolicyGradient(PolicyBase):
    def __init__(self, 
                    env,
                    batch_size,
                    device,
                    render,
                    lr,
                    n_epochs,
                    trained_model_path,
                    mlp_hidden_sizes):

        super().__init__(
                    env, 
                    batch_size,
                    device,
                    render,
                    lr,
                    n_epochs,
                    trained_model_path,
                    mlp_hidden_sizes)
        self.baseline_model = MLPPolicy(obs_dim=self.obs_dim, act_dim=1, layer_sizes=mlp_hidden_sizes).to(self.device)
        print("Baseline model: ", self.baseline_model)

    def train(self):
        # optimizer and things
        params = list(self.policy.parameters()) + list(self.baseline_model.parameters())
        if self.is_continuous:
            params.append(self.log_std)
        optimizer = optim.Adam(params, lr=self.lr)

        for epoch in range(self.n_epochs):
            self.policy.train()
            self.baseline_model.train()
            batch_obs, batch_acts, batch_weights, epoch_info = super().train()

            batch_baseline = self.baseline_model(batch_obs)
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

            loss = torch.mean(-batch_log_prob * (batch_weights - batch_baseline))
            mse = torch.mean((batch_weights - batch_baseline)**2)
            # because we want to minimize both loss, we can sum it
            final_loss = (loss + mse) / 2.
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print('Epoch {}, loss: {}, epoch info: {}'.format(epoch, final_loss,\
                        epoch_info))

        torch.save(self.best_policy_state_dict, self.trained_model_path)

    def perform(self):
        super().perform()
