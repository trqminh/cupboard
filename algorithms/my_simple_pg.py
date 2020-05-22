import gym
from gym.spaces import Box, Discrete
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions.categorical import Categorical
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
    hidden_sizes = configs['hidden_sizes']
    lr = float(configs['lr'])
    n_epochs = configs['n_epochs']
    batch_size = configs['batch_size']
    device = configs['device']
    render = configs['render']
    exp_name = configs['exp_name']

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    policy = mlp([obs_dim] + hidden_sizes + [n_acts]).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
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

                # batch_weights += ([ep_ret]*ep_len) # can ben reward to go
                batch_weights += list(reward_to_go(ep_rews))

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

        loss.backward(retain_graph=True)
        optimizer.step()

        saver_mean_ep_rets.append(np.mean(batch_rets))
        saver_mean_ep_lens.append(np.mean(batch_lens))

        if best_mean_episode_ret < np.mean(batch_rets):
            best_policy_state_dict = copy.deepcopy(policy.state_dict())

        if epoch % 10 == 0:
            print('Epoch {}, loss: {}, mean_episode_return: {:.1f}, mean_episode_len: {:.0f}'.format(epoch, loss,\
                    np.mean(batch_rets), np.mean(batch_lens)))


    policy.load_state_dict(best_policy_state_dict)
    torch.save(policy.state_dict(), configs['trained_model_path'])
    exp_path = './experiments/' + exp_name
    if not os.path.isdir(exp_path):
        os.mkdir(exp_path)
    np.savetxt(exp_path + '/' + configs['algo'] + '_mean_ep_ret.csv', np.asarray(saver_mean_ep_rets), delimiter=",")
    np.savetxt(exp_path + '/' + configs['algo'] + '_mean_ep_len.csv', np.asarray(saver_mean_ep_lens), delimiter=",")


def test(configs):
    if not os.path.isfile(configs['trained_model_path']):
        print('There is no trained model to validate')
        return None

    env = gym.make(configs['env'])
    hidden_sizes = configs['hidden_sizes']
    device = configs['device']

    obs = env.reset()
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    policy = mlp([obs_dim] + hidden_sizes + [n_acts]).to(device)
    policy.load_state_dict(torch.load(configs['trained_model_path']))
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

