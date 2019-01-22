#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


def parse_line(l):
    train = float(l.split(',')[2].split(':')[1].strip())
    test = float(l.split(',')[4].split(':')[1].strip())
    return train, test

def read_f(fn):
    train_noise = []
    test_noise = []
    for l in open(fn, 'r'):
        if l[0] != '[':
            continue
        train, test = parse_line(l)
        train_noise.append(train)
        test_noise.append(test)
    return train_noise, test_noise


def data():
    model = "./vgg16/"
    noise_level = ["0", "0.1", "0.2", "0.3"]
    level_color = ['firebrick', 'olivedrab', 'deepskyblue', 'darkorchid']
    dataf = [model + 'log_noise_{}.txt'.format(level) for level in noise_level]
    data = {}
    for i, level in enumerate(noise_level):
        data_level = {'train': [], 'test': []}
        f = dataf[i]
        train_noise, test_noise = read_f(f)
        data_level['train'] = train_noise
        data_level['test'] = test_noise
        data[level] = data_level

    for col, level in zip(level_color, noise_level):
        plt.plot(100 - 100 * np.array(data[level]['train']), color=col, linestyle='-', label='train, noise='+level)
        plt.plot(100 - 100 * np.array(data[level]['test']), color=col, linestyle='--', label='test, noise='+level)
    plt.legend()
    plt.ylim(0, 30)
    plt.show()

if __name__ == "__main__":
    data()
