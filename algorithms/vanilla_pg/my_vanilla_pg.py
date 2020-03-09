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


def train(env, hidden_size=32, lr=1e-2, n_epochs=500, batch_size=5000, device='cpu', render=False):
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    policy = mlp(obs_dim, n_acts, hidden_size).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    best_policy_state_dict = copy.deepcopy(policy.state_dict())
    best_mean_episode_ret = -1e6

    for epoch in range(n_epochs):
        policy.train()
        optimizer.zero_grad()

        batch_obs = []
        batch_acts = []
        batch_weights = []
        batch_rets = []
        batch_lens = []


        obs = env.reset()
        done = False
        ep_rews = []

        while True:
            #env.render()

            batch_obs.append(obs.copy())

            obs = torch.from_numpy(obs).to(dtype=torch.float, device=device)
            logit = policy(obs)
            m = Categorical(F.softmax(logit, dim=0))
            act = m.sample()

            obs, reward, done, info = env.step(act.item())


            batch_acts.append(act)
            ep_rews.append(reward)


            if done:
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)

                # batch_rets and batch_lens just for information
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                batch_weights += ([ep_ret]*ep_len) # can ben reward to go

                obs, done, ep_rews = env.reset(), False, []
                if len(batch_obs) > batch_size:
                    break


        batch_obs = torch.tensor(batch_obs).to(device)
        batch_acts = torch.tensor(batch_acts).to(device)
        batch_weights = torch.tensor(batch_weights).to(device)

        batch_logits = policy(batch_obs)
        batch_m = Categorical(F.softmax(batch_logits, dim=1))
        batch_log_prob = batch_m.log_prob(batch_acts)

        loss = torch.mean(-batch_log_prob * batch_weights)

        loss.backward()
        optimizer.step()

        if best_mean_episode_ret < np.mean(batch_rets):
            best_policy_state_dict = copy.deepcopy(policy.state_dict())

        if epoch % 10 == 0:
            print('Epoch {}, loss: {}, mean_episode_return: {:.1f}, mean_episode_len: {:.0f}'.format(epoch, loss,\
                    np.mean(batch_rets), np.mean(batch_lens)))


    policy.load_state_dict(best_policy_state_dict)
    torch.save(policy.state_dict(), './trained_models/my_vanilla_pg.pth')


def test(env, hidden_size=32, device='cpu', render=False):
    obs = env.reset()
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    policy = mlp(obs_dim, n_acts, hidden_size).to(device)
    policy.load_state_dict(torch.load('./trained_models/my_vanilla_pg.pth'))
    policy.eval()
    while True:
        env.render()
        obs = torch.from_numpy(obs).to(dtype=torch.float, device=device)

        logit = policy(obs)
        m = Categorical(F.softmax(logit, dim=0))
        act = m.sample()

        obs, reward, done, info = env.step(act.item())

        if done:
            print(reward)
            obs, done = env.reset(), False

