import csv
import numpy as np
import matplotlib.pyplot as plt
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='exp_1', help='experiment name')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    exp_name = args.exp_name

    exp = os.listdir('./experiments/' + exp_name + '/')
    subs = 'ret'
    ret_info = [i for i in info if subs in i]

    epochs = [i for i in range(1, 1001)]
    labels = ['vanilla_pg', 'simple_pg']

    for i, ret_file in enumerate(ret_info):
        rets = []
        with open('./info/' + ret_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                rets.append(float(row[0]))

        plt.plot(epochs, rets, label=labels[i])

    plt.ylabel('mean episode return')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.savefig('foo.png')
