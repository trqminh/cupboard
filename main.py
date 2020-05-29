from importlib import import_module
import gym
import argparse
import yaml
import torch
from algorithms import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/vpg_cont_config.yaml', help='config path')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    config_path = args.config
    configs = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)

    if configs['device'] == 'cuda:0':
        configs['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Configs: ')
    print(configs)

    env = gym.make(configs['env'])
    agent = getattr(import_module('algorithms'), configs['algo'])(configs, env)
    print('Agent: {}, Env: {}'.format(type(agent).__name__, env.unwrapped.spec.id))

    if configs['test']:
        agent.perform()
    else:
        agent.train()


if __name__ == '__main__':
    main()
