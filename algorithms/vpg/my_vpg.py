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


def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)

    return rtgs


def train(configs):
    env = gym.make(configs['env'])
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."

    is_continuous = isinstance(env.action_space, Box)
    hidden_sizes = configs['hidden_sizes']
    lr = float(configs['lr'])
    n_epochs = configs['n_epochs']
    batch_size = configs['batch_size']
    device = configs['device']
    render = configs['render']
    exp_name = configs['exp_name']

    # Declare policy
    obs_dim = env.observation_space.shape[0]
    out_layer_dim = None
    output_activation = None

    # log std for continuous
    log_std = None

    if is_continuous:
        out_layer_dim = env.action_space.shape[0]
        output_activation = nn.Tanh
        log_std = torch.tensor(-0.5*np.ones(out_layer_dim), dtype=torch.float32, requires_grad=True, device=device)
    else:
        out_layer_dim = env.action_space.n
        output_activation = nn.Identity

    policy = mlp(sizes = [obs_dim] + hidden_sizes + [out_layer_dim], output_activation=output_activation).to(device)
    baseline_model = mlp([obs_dim] + hidden_sizes + [1]).to(device)

    # optimizer and things
    params = list(policy.parameters())
    if is_continuous:
        params.append(log_std)

    optimizer = optim.Adam(params, lr=lr)
    optimizer_mse = optim.Adam(baseline_model.parameters(), lr=lr)
    best_policy_state_dict = copy.deepcopy(policy.state_dict())
    best_mean_episode_ret = -1e6

    saver_mean_ep_rets = []
    saver_mean_ep_lens = []

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
            if render:
                env.render()

            batch_obs.append(obs.copy()) # save the observation for offline update
            logit = policy(torch.from_numpy(obs).to(dtype=torch.float, device=device))

            if is_continuous:
                std = torch.exp(log_std)
                distribution = Normal(logit, std) # logit as mean
                act = distribution.sample()
            else:
                distribution = Categorical(F.softmax(logit, dim=0))
                act = distribution.sample()

            obs, reward, done, info = env.step(act.tolist())
            batch_acts.append(act.tolist())
            ep_rews.append(reward)

            if done:
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)

                # batch_rets and batch_lens just for information
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                batch_weights += list(reward_to_go(ep_rews)) # batch_weights += ([ep_ret]*ep_len)
                obs, done, ep_rews = env.reset(), False, []

                if len(batch_obs) > batch_size:
                    break

        batch_obs = torch.tensor(batch_obs).to(device)
        batch_acts = torch.tensor(batch_acts).to(device)
        batch_weights = torch.tensor(batch_weights).to(device)

        batch_baseline = baseline_model(batch_obs)
        batch_logits = policy(batch_obs)
        batch_distribution = None
        batch_log_prob = None

        if is_continuous:
            batch_log_std = torch.cat([log_std.unsqueeze(0)] * batch_logits.shape[0])
            batch_std = torch.exp(batch_log_std)
            batch_distribution = Normal(batch_logits, batch_std)
            batch_log_prob = batch_distribution.log_prob(batch_acts).sum(axis=-1)
        else:
            batch_distribution = Categorical(F.softmax(batch_logits, dim=1))
            batch_log_prob = batch_distribution.log_prob(batch_acts)

        loss = torch.mean(-batch_log_prob * (batch_weights - batch_baseline))
        loss.backward(retain_graph=True)
        optimizer.step()

        mse = torch.mean((batch_weights - batch_baseline)**2)
        mse.backward()
        optimizer_mse.step()

        saver_mean_ep_rets.append(np.mean(batch_rets))
        saver_mean_ep_lens.append(np.mean(batch_lens))

        if best_mean_episode_ret < np.mean(batch_rets):
            best_policy_state_dict = copy.deepcopy(policy.state_dict())

        if epoch % 10 == 0:
            print('[LOG STD]:\n', batch_log_std)
            print('Epoch {}, loss: {}, mean_episode_return: {:.1f}, mean_episode_len: {:.0f}'.format(epoch, loss,\
                    np.mean(batch_rets), np.mean(batch_lens)))


    policy.load_state_dict(best_policy_state_dict)
    torch.save(policy.state_dict(), configs['trained_model_path'])

    exp_path = './experiments/' + exp_name
    if not os.path.isdir(exp_path):
        os.mkdir(exp_path)
    np.savetxt(exp_path + '/' + configs['algo'] + '_mean_ep_ret.csv', np.asarray(saver_mean_ep_rets), delimiter=",")
    np.savetxt(exp_path + '/' + configs['algo'] + '_mean_ep_len.csv', np.asarray(saver_mean_ep_lens), delimiter=",")

