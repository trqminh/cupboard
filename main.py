from importlib import import_module
import gym
import argparse
import yaml
import torch
from algorithms import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/vpg_config.yaml', help='config path')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    config_path = args.config
    configs = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    print('Configs: ')
    print(configs)

    env = gym.make(configs['env'])
    if configs['device'] == 'cuda:0':
        configs['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    agent = getattr(import_module('algorithms'), configs['algo'])(configs, env)
    print('Agent: ', type(agent).__name__)
    if configs['test']:
        agent.perform()
    else:
        agent.train()


if __name__ == '__main__':
    main()
