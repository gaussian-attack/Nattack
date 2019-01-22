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

"""Builds the network.

Part of the source codes come from
http://tensorflow.org/tutorials/deep_cnn/

"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tensorflow as tf
from utils import IMG_SIZE, NUM_CHANNELS, NUM_CLASSES

flags = tf.app.flags
FLAGS = flags.FLAGS

from layers import *
from lenet import *
from resnet import *

# Global constants describing the CIFAR-10 data set.
if 'mnist' in FLAGS.data_set:
  NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 60000
  NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
elif 'cifar' in FLAGS.data_set:
  NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
  NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
elif 'imagenet' in FLAGS.data_set:
  NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1281167
  NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 50000

# for piecewice constant decay
if 'imagenet' in FLAGS.data_set:
  BOUNDARIES = [5000, 500000, 800000, 1000000, 1200000]
  LEARNING_RATES = [0.01, 0.1, 0.01, 0.001, 0.0001, 0.00001]
elif 'mnist' in FLAGS.data_set:
  BOUNDARIES = [4000, 6000, 8000]
  LEARNING_RATES = [0.1, 0.01, 0.001, 0.0001]
elif 'cifar' in FLAGS.data_set:
  BOUNDARIES = [400, 48000, 72000, 96000]
  LEARNING_RATES = [0.01, 0.1, 0.01, 0.001, 0.0001]


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 50      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.


# since this function will be called only for generating adversarial images
# we set the phase as False.
def step_forward(x, model_dict=None, reuse=True, main_scope_name='main'):
  """Build the Inference model for adversarial images generation.

  Args:
    x: input 4D tensor of shape [num_batch, width, height, channel]
    model_dict: dictionary which contains parameters for model architecture.
    reuse: True to reuse the tensor variables to define graph
    main_scope_name: String for the scope name

  Returns:
    logits: Logits. 2D tensor of shape [num_batch, num_classes]
  """
  images = tf.reshape(x, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])
  # interence
  embeddings, logits = inference(
      images=images,
      phase=tf.constant(False),
      is_for_adver=tf.constant(True),
      model_dict=model_dict,
      reuse=reuse,
      main_scope_name=main_scope_name)
  return logits

def step_backward(x, logits, labels, main_scope_name='main'):
  """Build gradient graph for adversarial images generation.

  Args:
    x: input 4D tensor of shape [num_batch, width, height, channel]
    logits: Logits. 2D tensor of shape [num_batch, num_classes]
    labels: True labels. 2D tensor of shape [num_batch, num_classes]
    main_scope_name: String for the scope name

  Returns:
    grad_x: gradient. 4D tensor of shape [num_batch, width, height, channel]
  """
  loss = _sparse_cross_entropy_loss(logits, labels,
                            1., main_scope_name + 'step_cross_entropy',
                            True, 'step_losses')
  [grad_x] = tf.gradients(loss, x)
  return grad_x

def _per_image_standardization(images, flag):
  """Returns images tensor after standardization per image.

  Args:
    images: images. 4D tensor of shape [num_batch, width, height, channel]
    flag: True: standardization, False: bypass the network.

  Returns:
    result: standardized images.
            4D tensor of shape [num_batch, width, height, channel]
  """
  with tf.variable_scope('standardization') as scope:
    if flag:
      result = tf.map_fn(
          lambda img: tf.image.per_image_standardization(img), images)
    else:
      result = images
  return result

def inference(images, phase, is_for_adver,
              model_dict=None,
              reuse=False,
              main_scope_name='main'):
  """Build the Inference model.

  Args:
    images: Images, 4D tensor of shape [num_batch, width, height, channel]
    phase: bool tensor (True: training, False: test) 
    is_for_adver: bool tensor (True: inference for clean image,
                  False: inference for adversarial image generation)
    model_dict: dictionary which contains parameters for model architecture.
    reuse: True to reuse the tensor variables to define graph
    main_scope_name: String for the scope name

  Returns:
    Embeddings and Logits.
  """
  if model_dict is not None:
    per_image_standard = model_dict['per_image_standard']
    add_noise_inference = model_dict['add_noise_inference']
    model_name = model_dict['model_name']
    resnet_n = model_dict['resnet_n']
    embedding_at = model_dict['embedding_at']
  else:
    per_image_standard = FLAGS.per_image_standard
    add_noise_inference = FLAGS.add_noise_inference
    model_name = FLAGS.model_name
    resnet_n = FLAGS.resnet_n
    embedding_at = FLAGS.embedding_at

  # Pre-processing, we do this as part of the inference
  # To avoid any mistake to generate adversarial images
  images = _per_image_standardization(images, per_image_standard)

# ==============================================================================
#  Add noise or not
# ==============================================================================
  with tf.variable_scope(main_scope_name, reuse=reuse) as scope:
    if add_noise_inference:
      noisy_images = images + \
          (16/255.)*tf.sign(tf.random_normal(images.get_shape()[1:]))
      # add noise only for inference at test time
      # for training and adversarial image generation, we don't add noise.
      images = tf.cond(tf.logical_or(phase, is_for_adver),
          lambda: images, lambda: noisy_images)
# ==============================================================================
#  Main CNN 
# ==============================================================================
    if model_name == 'lenet':
      embedding, logit = lenet(
          images, IMG_SIZE, NUM_CHANNELS, NUM_CLASSES, embedding_at)
    elif model_name == 'resnet':
      embedding, logit = resnet(images, IMG_SIZE, NUM_CHANNELS, NUM_CLASSES,
          phase, resnet_n, embedding_at)
    elif model_name == 'resnet_e2':
      embedding, logit = resnet_e2(images, IMG_SIZE, NUM_CHANNELS, NUM_CLASSES,
          phase, resnet_n, embedding_at)
    elif model_name == 'resnet_e10x2':
      embedding, logit = resnet_e10x2(images, IMG_SIZE, NUM_CHANNELS,
          NUM_CLASSES, phase, resnet_n, embedding_at)
    elif model_name == 'noresnet':
      embedding, logit = noresnet(images, IMG_SIZE, NUM_CHANNELS, NUM_CLASSES,
          phase, resnet_n, embedding_at)
    else:
      raise ValueError('Please choose between resnet and lenet')

  # shape of logit = [batch_size, NUM_CLASSES]
  return embedding, logit

def _get_similarity_loss(embeddings, num_same_images, offset, factor,
                         collection_name=None):
  with tf.variable_scope('similarity') as scope:
    diff = embeddings[0:num_same_images] \
           - embeddings[offset:offset+num_same_images]
    if FLAGS.distance == 'l1':
      similarity_loss = tf.multiply(tf.reduce_sum(tf.abs(diff)),
                                    factor, name='similarity_loss')
    elif FLAGS.distance == 'l2':
      similarity_loss = tf.multiply(tf.nn.l2_loss(diff),
                                    factor, name='similarity_loss')
    else:
      raise ValueError('Please choose between l1 and l2 for FLAGS.distance')

  tf.add_to_collection(collection_name, similarity_loss)
  return similarity_loss

def _get_pivot_loss(embeddings, num_same_images, offset, factor,
                    collection_name=None):
  with tf.variable_scope('pivot') as scope:
    nrof_embeddings = embeddings.get_shape()[1]
    # We define pivots as non-trainable variables to make sure
    # only the adversarial mini-batches are trained to minimize the distance
    # between the clean mini-batches
    pivots = tf.get_variable('pivot', [num_same_images, nrof_embeddings],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0),
                             trainable=False)
    # Update pivots with embeddings from the clean images
    # We assume [0:num_same_images] embeddings from the adversarial images 
    # We assume [offset:offset+num_same_images] embeddings from the clean images
    if FLAGS.normalize_embedding:
      pivots = embeddings[offset:offset+num_same_images]
      mean, var = tf.nn.moments(pivots, axes=[1])
      normalized_pivots = pivots/var[:, None]
      diff = embeddings[0:num_same_images]/var[:, None] - normalized_pivots
    else:
      pivots = embeddings[offset:offset+num_same_images]
      diff = embeddings[0:num_same_images] - pivots
    if FLAGS.distance == 'l1':
      pivot_loss = tf.multiply(tf.reduce_sum(tf.abs(diff)),
                               factor, name='pivot_loss')
    elif FLAGS.distance == 'l2':
      pivot_loss = tf.multiply(tf.nn.l2_loss(diff), factor, name='pivot_loss')
    else:
      raise ValueError('Please choose between l1 and l2 for FLAGS.distance')
  tf.add_to_collection(collection_name, pivot_loss)
  return pivot_loss

def _sparse_cross_entropy_loss(logits, labels, factor, name, add_to_collection,
                              collection_name=None,
                              offset=0, num_same_images=0,
                              adver_loss_factor=1., samebatch_loss_factor=1.):
  with tf.variable_scope(name) as scope:
    labels = tf.cast(labels, tf.int64)
    batch = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name=name + '_per_example')
    batch = tf.scalar_mul(factor, batch)
    if offset > 0: # if adversarial training
      # adversarial examples from 0 to offset
      temp_0 = adver_loss_factor * tf.reduce_sum(batch[:offset])
      if num_same_images > 0: # if similarity or pivot loss is used
        temp_1 = samebatch_loss_factor * \
                 tf.reduce_sum(batch[offset:offset+num_same_images])
        if offset+num_same_images < FLAGS.batch_size:
          temp_1 = temp_1 + tf.reduce_sum(batch[offset+num_same_images:])
      else:
        temp_1 = tf.reduce_sum(batch[offset:])
      cross_entropy = tf.multiply(temp_0+temp_1,
                      1/(FLAGS.batch_size-offset-num_same_images
                         +samebatch_loss_factor*num_same_images
                         +adver_loss_factor*offset),
                      name=name)
    else:
      cross_entropy = tf.reduce_mean(batch, name=name)
  if add_to_collection:
    tf.add_to_collection(collection_name, cross_entropy)
  return cross_entropy


def loss(logits, labels, embeddings):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """

  adver_batch_size = FLAGS.adver_batch_size if FLAGS.adversarial else 0
  if FLAGS.similarity_loss_factor > 0.0 or FLAGS.pivot_loss_factor > 0.0:
    num_same_images = min(FLAGS.adver_batch_size,
                          FLAGS.batch_size-FLAGS.adver_batch_size)
  else:
    num_same_images = 0
  # Calculate the average cross entropy loss across the batch.
  cross_entropy = _sparse_cross_entropy_loss(logits, labels,
                                          1., 'cross_entropy',
                                          True, 'losses',
                                          adver_batch_size, num_same_images,
                                          FLAGS.adver_loss_factor,
                                          FLAGS.samebatch_loss_factor)

  num_same_images = min(FLAGS.adver_batch_size,
                        FLAGS.batch_size-FLAGS.adver_batch_size)
  offset = FLAGS.adver_batch_size
  # Similarity loss
  similarity_loss =  _get_similarity_loss(embeddings, num_same_images, offset,
                                         FLAGS.similarity_loss_factor,
                                         'losses')
  # Pivot loss
  pivot_loss =  _get_pivot_loss(embeddings, num_same_images, offset,
                               FLAGS.pivot_loss_factor,
                               'losses')
  similarity_loss = similarity_loss + pivot_loss

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return cross_entropy, similarity_loss

def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step, verbose=False):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.piecewise_constant(tf.to_int32(global_step),
                                   BOUNDARIES, LEARNING_RATES)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)
#  print (total_loss)
#  print (loss_averages_op)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  t_vars = tf.trainable_variables()
  main_vars  = [var for var in t_vars if 'main' in var.name]
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.MomentumOptimizer(lr, 0.9)

    main_grads = tf.gradients(total_loss, main_vars)

  apply_gradient_op = opt.apply_gradients(zip(main_grads, main_vars),
                                          global_step=global_step)

  # Add histograms for gradients.
  if verbose:
    print ('Gradients')
  for grad in main_grads:
    if grad is not None:
      if verbose:
        print (grad)
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  # Collect update ops
  update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  if update_op:
    updates = tf.tuple(update_op)
    print ("Update ops are:")
    for update in updates:
      print (update)
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]
                                 + update_op):
      train_op = tf.no_op(name='train')
  else:
    print ("There is no update ops")
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
      train_op = tf.no_op(name='train')

  return train_op, lr


