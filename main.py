#import algorithms.VanillaPG.my_vanilla_pg as my_vpg
import importlib
import gym
import argparse
import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/vpg_config.yaml', help='config path')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    config_path = args.config
    configs = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)

    algo_name = configs['algo']
    my_module = importlib.import_module('algorithms.' + algo_name + '.my_' + algo_name)
    env = gym.make(configs['env'])

    if configs['test']:
        my_module.test(configs)
    else:
        my_module.train(configs)


if __name__ == '__main__':
    main()
