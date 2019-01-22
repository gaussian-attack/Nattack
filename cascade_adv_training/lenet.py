# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
import tensorflow as tf

from layers import *

def lenet(images, img_size, num_channels, num_classes,
          embedding_at=2):
  # conv1
  with tf.variable_scope('gated_conv1') as scope:
    pre_activation = gated_conv(images, 5, 5, num_channels, 64, None)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    activation_summary(conv1)
  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  if embedding_at == 1:
    nrof_embeddings = img_size*img_size*16
    embedding = tf.reshape(norm1, [-1, nrof_embeddings])

  # conv2
  with tf.variable_scope('gated_conv2') as scope:
    pre_activation = gated_conv(norm1, 5, 5, 64, 64, None)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    activation_summary(conv2)
  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  assert pool2.get_shape().as_list()[1:] == [img_size/4, img_size/4, 64]

  dim_unroll = img_size*img_size*4
  reshape = tf.reshape(pool2, [-1, dim_unroll])
  if embedding_at == 2:
    embedding = reshape

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    pre_activation = fc(reshape,
                        dim_unroll, 384, 0.04, 0.004, 
                        'pre_' + scope.name)
    local3 = tf.nn.relu(pre_activation, name=scope.name)
    activation_summary(local3)

  if embedding_at == 3:
    embedding = local3

  # local4
  with tf.variable_scope('local4') as scope:
    pre_activation = fc(local3,
                        384, 192, 0.04, 0.004, 
                        'pre_' + scope.name)
    local4 = tf.nn.relu(pre_activation, name=scope.name)
    activation_summary(local4)

  if embedding_at == 4:
    embedding = local4

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('logit') as scope:
    logit = fc(local4,
               192, num_classes, 1/192.0, 0.0, 
               scope.name)
    activation_summary(logit)

  if embedding_at > 4:
    embedding = logit

  return embedding, logit

