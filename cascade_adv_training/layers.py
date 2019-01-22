
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

TOWER_NAME = 'tower'


def variable_on_cpu(name, shape, initializer, trainable=True):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
#  with tf.device('/cpu:0'):
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = tf.get_variable(name, shape, initializer=initializer,
                        dtype=dtype, trainable=trainable)
  return var


def variable_with_weight_decay(name, shape, stddev, wd,
                               collection_name='losses', initializer=None, trainable=True):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  if initializer is not None:
    var = variable_on_cpu(name, shape, initializer, trainable)
  else:
    var = variable_on_cpu(name, shape,
                          tf.contrib.layers.xavier_initializer(), trainable)
#      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype), trainable)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection(collection_name, weight_decay)
  return var

def activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))

def fc(input, num_inputs, num_outputs, stddev, wd, name, collection_name='losses'):
  weights = variable_with_weight_decay('weights', shape=[num_inputs, num_outputs],
                                        stddev=stddev, wd=wd, collection_name=collection_name)
  biases = variable_on_cpu('biases', [num_outputs], tf.constant_initializer(0.1))
  pre_activation = tf.add(tf.matmul(input, weights), biases, name=name)
  return pre_activation

def conv(input, width, height, depth, num_kernels, collection_name='losses'):
  kernel = variable_with_weight_decay('weights',
                                       shape=[width, height, depth, num_kernels],
                                       stddev=5e-2,
                                       wd=0.0, collection_name=collection_name)
  conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
  biases = variable_on_cpu('biases', [num_kernels], tf.constant_initializer(0.0))
  pre_activation = tf.nn.bias_add(conv, biases)
  return pre_activation

def gated_conv(input, width, height, depth, num_kernels, softmax_style=None):
  kernel0 = variable_with_weight_decay('weights0',
            shape=[width, height, depth, num_kernels],
            stddev=5e-2, wd=0.0)
  biases0 = variable_on_cpu('biases0', [num_kernels],
            tf.constant_initializer(0.1))

  if (softmax_style is not None):
#    gated_results = []
#    for i in range(batch_size):
#      gated_kernel = tf.add(softmax_style[i,0]*kernel[0],
#                            softmax_style[i,1]*kernel[1])
#      gated_biases = tf.add(softmax_style[i,0]*biases[0],
#                            softmax_style[i,1]*biases[1])
#      conv_0 = tf.nn.conv2d(tf.expand_dims(input[i], 0), gated_kernel,
#                            [1, 1, 1, 1], padding='SAME')
#      pre_activation_0 = tf.nn.bias_add(conv_0, gated_biases)
#      gated_results.append(pre_activation_0)
#    pre_activation = tf.concat(gated_results, 0)

    kernel1 = variable_with_weight_decay('weights1',
              shape=None,
              stddev=5e-2, wd=0.0,
              initializer=kernel0.initialized_value())
    biases1 = variable_on_cpu('biases1', [num_kernels],
              tf.constant_initializer(0.1))

    conv_0 = tf.nn.conv2d(input, kernel0, [1, 1, 1, 1], padding='SAME')
    pre_activation_0 = tf.nn.bias_add(conv_0, biases0)
    conv_1 = tf.nn.conv2d(input, kernel1, [1, 1, 1, 1], padding='SAME')
    pre_activation_1 = tf.nn.bias_add(conv_1, biases1)
    pre_activation =  tf.add(softmax_style[:,0,None,None,None]*pre_activation_0,
                                 softmax_style[:,1,None,None,None]*pre_activation_1)
  else:
    conv = tf.nn.conv2d(input, kernel0, [1, 1, 1, 1], padding='SAME')
    pre_activation = tf.nn.bias_add(conv, biases0)
  return pre_activation

