import csv
import numpy as np
import matplotlib.pyplot as plt
import os

info = os.listdir('./info/')
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
