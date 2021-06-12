import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='exp_1', help='experiment name')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    exp_name = args.exp_name
    exp_path = './experiments/' + exp_name + '/'

    exp_result_files = os.listdir(exp_path)
    for item in ['ret', 'len']:
        file_names = [i for i in exp_result_files if item in i]
        epochs = [i for i in range(1, 1001)]

        for file_name in file_names:
            rets = []
            ext = os.path.splitext(file_name)[1]
            if ext != '.csv':
                continue

            with open(exp_path + file_name, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    rets.append(float(row[0]))

            label = os.path.splitext(file_name)[0]
            plt.plot(epochs, rets, label=label)

        plt.ylabel('mean episode ' + item)
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        plt.savefig(exp_path + item + '.png')
        plt.clf()


if __name__ == '__main__':
    main()
