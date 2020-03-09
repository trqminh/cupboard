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
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    
    algo_name = config['algo']
    module = importlib.import_module('algorithms.' + algo_name + '.my_' + algo_name)
    env = gym.make(config['env'])

    if config['validate']:
        module.test(env)



if __name__ == '__main__':
    main()
