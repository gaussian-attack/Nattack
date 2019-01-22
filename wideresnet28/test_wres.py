from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import scipy.misc
from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer
from discretization_utils import discretize_uniform
from discretization_attacks import adv_lspga

import tensorflow as tf
import numpy as np

from cifar_model import Model
import cifar10_input
import sys
import robustml
import argparse

levels = 16






steps = 7
eps = 0.031



parser = argparse.ArgumentParser()
parser.add_argument('--cifar-path', type=str, default='../cifar10_data/test_batch',
        help='path to the test_batch file from http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=100)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()


provider = robustml.provider.CIFAR10(args.cifar_path)
# saver = tf.train.Saver(max_to_keep=3)

start = 0
end = 100

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

model = Model( './models/adv_train_clean/',
            sess,mode='eval', tiny=False,
              thermometer=False, levels=levels)
  # initialize data augmentation


input_xs = tf.placeholder(tf.float32, [None, 32, 32, 3])
real_logits = tf.nn.softmax(model(input_xs))

# saver.restore(sess,
#             os.path.join("models/adv_train_clean", 'checkpoint-65000'))

for i in range(start, end):
    x_batch, y_batch = provider[i]

    logits = sess.run(real_logits,feed_dict={input_xs: [x_batch]})
#     nat_dict = {model.x_input: [x_batch],model.y_input: [y_batch]}
#     logits = sess.run(model.pre_softmax, feed_dict=nat_dict)

    print(logits)
    
    


