import sys
sys.path.append('./')


from importlib import import_module
import gym
import argparse
import yaml
import torch
from cupboard.algos.pg.my_simple_pg import SimplePolicyGradient


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/simple_pg_config.yaml',
                        help='config path')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    config_path = args.config
    configs = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)

    print('Configs: ')
    print(configs)

    env = gym.make(configs['env'])
    agent = SimplePolicyGradient(env, 
                                 configs['batch_size'],
                                 configs['device'],
                                 configs['render'],
                                 float(configs['lr']),
                                 configs['n_epochs'],
                                 configs['trained_model_path'])

    print('Agent: {}, Env: {}'.format(type(agent).__name__, env.unwrapped.spec.id))

    if configs['test']:
        agent.perform()
    else:
        agent.train()


if __name__ == '__main__':
    main()
