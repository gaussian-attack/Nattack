import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

WEIGHT_DECAY = 0.0001

from layers import variable_on_cpu
from layers import variable_with_weight_decay

def fc_layer(inpt, shape, wd=WEIGHT_DECAY, collection_name="losses"):
  fc_w = variable_with_weight_decay('weights', shape, stddev=0.1, wd=wd,
                                    collection_name=collection_name)
  fc_b = variable_on_cpu('biases', [shape[1]], tf.constant_initializer(0.))

  fc_h = tf.matmul(inpt, fc_w) + fc_b

  return fc_h

def softmax_layer(inpt, shape, wd=WEIGHT_DECAY, collection_name="losses"):
  fc_h = tf.nn.softmax(fc_layer(inpt, shape, wd, collection_name))

  return fc_h

def base_conv_layer(inpt, filter_shape, stride, wd=WEIGHT_DECAY,
                    collection_name="losses"):
  out_channels = filter_shape[3]

  filter_ = variable_with_weight_decay('weights', filter_shape,
                                       stddev=0.1, wd=wd,
                                       collection_name=collection_name)
  conv = tf.nn.conv2d(inpt, filter=filter_,
                      strides=[1, stride, stride, 1], padding="SAME")

  return conv

def conv_layer(inpt, filter_shape, stride, phase,
               wd=WEIGHT_DECAY, collection_name="losses"):
  out_channels = filter_shape[3]

  filter0_ = variable_with_weight_decay('weights0', filter_shape,
                                        stddev=0.1, wd=wd,
                                        collection_name=collection_name)
  conv = tf.nn.conv2d(inpt, filter=filter0_,
                      strides=[1, stride, stride, 1], padding="SAME")

  bn = tf.contrib.layers.batch_norm(conv, center=True, scale=True,
                                    is_training=phase, updates_collections=None, scope='bn')
  return bn

def gated_conv_layer(inpt, filter_shape, stride, phase,
                     wd=WEIGHT_DECAY, collection_name="losses"):
  out_channels = filter_shape[3]

  filter0_ = variable_with_weight_decay('weights0', filter_shape,
                                        stddev=0.1, wd=wd,
                                        collection_name=collection_name)
  conv = tf.nn.conv2d(inpt, filter=filter0_,
                      strides=[1, stride, stride, 1], padding="SAME")

  bn = tf.contrib.layers.batch_norm(conv, center=True, scale=True,
                                    is_training=phase, updates_collections=None, scope='bn')
  return bn


def noresidual_block(inpt, output_depth, down_sample,
                     phase, wd=WEIGHT_DECAY,
                     collection_name="losses"):
  input_depth = inpt.get_shape().as_list()[3]

  # conv path
  with tf.variable_scope('conv1') as scope:
    stride = 2 if down_sample else 1
    conv1 = conv_layer(inpt, [3, 3, input_depth, output_depth], stride,
                       phase, wd, collection_name=collection_name)
    conv1 = tf.nn.relu(conv1)
  with tf.variable_scope('conv2') as scope:
    conv2 = conv_layer(conv1, [3, 3, output_depth, output_depth], 1,
                       phase, wd, collection_name=collection_name)

  res = tf.nn.relu(conv2)
  return res

def residual_block(inpt, output_depth, down_sample,
                   phase, projection=False, wd=WEIGHT_DECAY,
                   collection_name="losses"):
  input_depth = inpt.get_shape().as_list()[3]

  # conv path
  with tf.variable_scope('conv1') as scope:
    stride = 2 if down_sample else 1
    conv1 = conv_layer(inpt, [3, 3, input_depth, output_depth], stride,
                       phase, wd,collection_name=collection_name)
    conv1 = tf.nn.relu(conv1)
  with tf.variable_scope('conv2') as scope:
    conv2 = conv_layer(conv1, [3, 3, output_depth, output_depth], 1,
                       phase, wd, collection_name=collection_name)

  # residual path
  with tf.variable_scope('residual') as scope:
    if down_sample:
      filter_ = [1,2,2,1]
      inpt1 = tf.nn.avg_pool(inpt, ksize=filter_, strides=filter_, padding='SAME')
#      inpt1 = tf.nn.max_pool(inpt, ksize=filter_, strides=filter_, padding='SAME')
    else:
      inpt1 = inpt

  # projection
  with tf.variable_scope('projection') as scope:
    if input_depth != output_depth:
      if projection:
        # Option B: Projection shortcut
        input_layer = base_conv_layer(inpt1, [1, 1, input_depth, output_depth], 1, wd,
                                 collection_name=collection_name)
      else:
        # Option A: Zero-padding
        input_layer = tf.pad(inpt1, [[0,0], [0,0], [0,0], [0, output_depth - input_depth]])
    else:
      input_layer = inpt1

  res = conv2 + input_layer
  res = tf.nn.relu(res)
  return res

def gated_residual_block(inpt, output_depth, down_sample,
                         phase,
                         projection=False, wd=WEIGHT_DECAY,
                         collection_name="losses"):
  input_depth = inpt.get_shape().as_list()[3]

  # conv path
  with tf.variable_scope('conv1') as scope:
    stride = 2 if down_sample else 1
    conv1 = gated_conv_layer(inpt, [3, 3, input_depth, output_depth], stride,
                             phase, wd,
                             collection_name=collection_name)
    conv1 = tf.nn.relu(conv1)
  with tf.variable_scope('conv2') as scope:
    conv2 = gated_conv_layer(conv1, [3, 3, output_depth, output_depth], 1,
                             phase, wd,
                             collection_name=collection_name)

  # residual path
  with tf.variable_scope('residual') as scope:
    if down_sample:
      filter_ = [1,2,2,1]
      inpt1 = tf.nn.avg_pool(inpt, ksize=filter_, strides=filter_, padding='SAME')
#      inpt1 = tf.nn.max_pool(inpt, ksize=filter_, strides=filter_, padding='SAME')
    else:
      inpt1 = inpt

  # projection
  with tf.variable_scope('projection') as scope:
    if input_depth != output_depth:
      if projection:
        # Option B: Projection shortcut
        input_layer = base_conv_layer(inpt1, [1, 1, input_depth, output_depth], 1, wd,
                                      collection_name=collection_name)
      else:
        # Option A: Zero-padding
        input_layer = tf.pad(inpt1, [[0,0], [0,0], [0,0], [0, output_depth - input_depth]])
    else:
      input_layer = inpt1

  res = conv2 + input_layer
  res = tf.nn.relu(res)
  return res
 
def residual_bottleneck(inpt, output_depth, down_sample,
                        phase, projection=True, wd=WEIGHT_DECAY,
                        collection_name="losses"):
  input_depth = inpt.get_shape().as_list()[3]

  # conv path
  with tf.variable_scope('conv1') as scope:
    stride = 2 if down_sample else 1
    conv1 = conv_layer(inpt, [1, 1, input_depth, output_depth/4], stride,
                       phase, wd, collection_name=collection_name)
    conv1 = tf.nn.relu(conv1)
  with tf.variable_scope('conv2') as scope:
    conv2 = conv_layer(conv1, [3, 3, output_depth/4, output_depth/4], 1,
                       phase, wd, collection_name=collection_name)
    conv2 = tf.nn.relu(conv2)
  with tf.variable_scope('conv3') as scope:
    conv3 = conv_layer(conv2, [1, 1, output_depth/4, output_depth], 1,
                       phase, wd, collection_name=collection_name)

  # residual path
  with tf.variable_scope('residual') as scope:
    if down_sample:
      filter_ = [1,2,2,1]
      inpt1 = tf.nn.avg_pool(inpt, ksize=filter_, strides=filter_, padding='SAME')
#      inpt1 = tf.nn.max_pool(inpt, ksize=filter_, strides=filter_, padding='SAME')
    else:
      inpt1 = inpt

  # projection
  with tf.variable_scope('projection') as scope:
    if input_depth != output_depth:
      if projection:
        # Option B: Projection shortcut
        input_layer = base_conv_layer(inpt1, [1, 1, input_depth, output_depth], 1, wd,
                                      collection_name=collection_name)
      else:
        # Option A: Zero-padding
        input_layer = tf.pad(inpt1, [[0,0], [0,0], [0,0], [0, output_depth - input_depth]])
    else:
      input_layer = inpt1

  res = conv3 + input_layer
  res = tf.nn.relu(res)
  return res

