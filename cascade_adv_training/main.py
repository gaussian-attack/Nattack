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

"""This is to reproduce the results reported in the paper
(Taesik Na et. al 2018)
Paper link: https://openreview.net/forum?id=HyRVBzap-

Part of the source code comes from
http://tensorflow.org/tutorials/deep_cnn/

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import math

import pprint
import numpy as np
import tensorflow as tf
import os

try:
  image_summary = tf.image_summary
  scalar_summary = tf.scalar_summary
  histogram_summary = tf.histogram_summary
  merge_summary = tf.merge_summary
  SummaryWriter = tf.train.SummaryWriter
except:
  image_summary = tf.summary.image
  scalar_summary = tf.summary.scalar
  histogram_summary = tf.summary.histogram
  merge_summary = tf.summary.merge
  SummaryWriter = tf.summary.FileWriter

pp = pprint.PrettyPrinter()
flags = tf.app.flags

# Define Dataset and Model
flags.DEFINE_string('data_set', 'cifar10',
                    """Choose between mnist, cifar10.""")
flags.DEFINE_string('model_name', 'resnet',
                    """Name of CNN model"""
                    """Choose between lenet and resnet.""")
flags.DEFINE_integer('resnet_n', 18,
                     """n for RESNET {3:20, 5:32, 7:44, 9:56, 18:110}.""")
flags.DEFINE_integer("embedding_at", 5,
                     """position of embedding [2]""")

# Training Mode
flags.DEFINE_boolean("is_train", False,
                     """True for training, False for testing [False]""")
flags.DEFINE_boolean("restore", False,
                     """True for restoring variables from the checkpoint_dir"""
                     """ [False]""")
flags.DEFINE_boolean("restore_inplace", False,
                     """True for restoring variables from the train_dir"""
                     """ [False]""")
flags.DEFINE_boolean("use_saved_images", False,
                     """True for testing with saved_images stored in"""
                     """ FLAGS.saved_dir [False]""")
flags.DEFINE_string('saved_data_dir', '/tmp/mnist_train',
                    """Directory where to load adversarial images for test.""")
flags.DEFINE_boolean("cascade", False,
                     """True for using saved iter_fgsm images for training"""
                     """ [False]""")
flags.DEFINE_boolean("ensemble", False,
                     """True for ensemble training"""
                     """ [False]""")
flags.DEFINE_string('saved_iter_fgsm_dir', '/tmp/mnist_train',
                    """Directory which contains iter_fgsm images"""
                    """ for training.""")
flags.DEFINE_boolean('log_device_placement', False,
                     """Whether to log device placement.""")
flags.DEFINE_boolean('add_noise_inference', False,
                     """True for adding random noise for inference.""")


# Define Directories
flags.DEFINE_string('data_dir', './data',
                    """Path to the CIFAR-10 data directory.""")
flags.DEFINE_string('eval_dir', '/tmp/mnist_eval',
                    """Directory where to write event logs.""")
flags.DEFINE_string('eval_data', 'test',
                    """Either 'test' or 'train_eval'.""")
flags.DEFINE_string('checkpoint_dir', '/tmp/mnist_train',
                    """Directory where to read model checkpoints.""")
flags.DEFINE_string('train_dir', '/tmp/mnist_train',
                    """Directory where to write event logs """
                    """and checkpoint.""")

# Training Hyper Parameters
flags.DEFINE_integer('max_epochs', 240,
                     """Number of batches to run.""")
flags.DEFINE_integer('batch_size', 128,
                     """Number of images to process in a batch.""")
flags.DEFINE_integer('adver_batch_size', 64,
                     """Number of adversarial images to process in a batch.""")
flags.DEFINE_float('max_e', 16,
                   """Maximum magnitude for adversarial noise""")
flags.DEFINE_boolean('use_fp16', False,
                     """Train the model using fp16.""")

# Loss Parameters
flags.DEFINE_float("samebatch_loss_factor", 1.,
                   """samebatch loss factor [0.5]""")
flags.DEFINE_float("adver_loss_factor", 0.3,
                   """adver loss factor [0.3]""")
flags.DEFINE_float("similarity_loss_factor", 0.0,
                   """similarity loss factor [0.0001]""")
flags.DEFINE_float("pivot_loss_factor", 0.0,
                   """pivot loss factor [0.0001]""")
flags.DEFINE_string('distance', 'l2',
                    """Choose between l1 and l2.""")
flags.DEFINE_boolean('normalize_embedding', False,
                     """True for embedding normalization.""")

# Input preprocessing, adversarial training or not
flags.DEFINE_integer('norm_option', 0,
                     """0: per pixel, 1: per channel, 2: no normalization""")
flags.DEFINE_boolean("per_image_standard", False,
                     """True for per_image_standardization [False]""")
flags.DEFINE_boolean("rand_crop", True,
                     """True for random crop for inputs [True]""")
flags.DEFINE_boolean("adversarial", True,
                     """True for Adversarial training [True]""")
flags.DEFINE_integer('adver_option', 2,
                     """0: max_e, 1: uniform random, """
                     """2: truncated normal, 3: reverse truncated normal""")
flags.DEFINE_integer('sparsity', 100,
                     """represents the portion of non-zeros in adver_noise """)
flags.DEFINE_boolean("per_pixel_rand", False,
                     """True for per pixel random e for adversarial training"""
                     """ [False]""")
flags.DEFINE_string('method', 'step_ll',
                    """Choose between step_ll, step_fgsm, step_both and """
                    """ step_rand.""")

# For analysis
flags.DEFINE_boolean("save_adver_images", False,
                     """True for saving adversarial images during test."""
                     """Saved images can be used for transfer analysis.""")
flags.DEFINE_boolean("save_iter_fgsm_images", False,
                     """True for saving iter_fgsm adversarial images."""
                     """This is for cascade adversarial training.""")
flags.DEFINE_string("test_data_from", 'validation',
                    """Choose between train, validation and test""")


FLAGS = flags.FLAGS

# After setting all the parameters, load necessary functions
# for training and testing
from utils import *
import model

# ==============================================================================
# Settings
# ==============================================================================
# if you want to perform additional analysis
print_tensors = False
show_flag = False
show_cm_flag = False
print_accuracy = True

show_analysis_flag = True
analyze_corr_grads_flag = True
visualize_embeddings_flag = True
trace_embeddings_flag = True

save_adver_images = FLAGS.save_adver_images
save_iter_fgsm_images = FLAGS.save_iter_fgsm_images
test_data_from = FLAGS.test_data_from

test_e = []
methods = []
sparsities = []
if print_accuracy or save_adver_images:
  test_e = [0, 2, 4, 8, 16]
  methods = ['step_ll', 'step_fgsm', 'step_rand', 'iter_ll', 'iter_fgsm']
  sparsities = [100]
if save_iter_fgsm_images:
  save_adver_images = True
  test_e = [i for i in range(17)]
  methods = ['iter_fgsm']
  sparsities = [100]


np.random.seed(1234)
test_idx = np.random.choice(10000, 128)

# ==============================================================================
# Read Inputs
# ==============================================================================

def save(sess, saver, checkpoint_dir, step):
  """Saves tensorflow checkpoint.

  Args:
    sess: tensorflow session
    saver: tensorflow saver object
    checkpoint_dir: directory where the checkpoint is saved
    step: global step
              
  """
  model_name = "anti_adver.model"
  checkpoint_dir = os.path.join(checkpoint_dir)

  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

  saver.save(sess,
          os.path.join(checkpoint_dir, model_name),
          global_step=step)

def optimistic_restore(sess, save_file):
  """Restore the variables which match with the variables in current graph.
     This function is from RalphMao commented on 17 Mar
     at https://github.com/tensorflow/tensorflow/issues/312

  Args:
    sess: tensorflow session
    save_file: tensorflow checkpoint path
              
  """
  reader = tf.train.NewCheckpointReader(save_file)
  saved_shapes = reader.get_variable_to_shape_map()
  var_names = sorted([(var.name, var.name.split(':')[0])
                    for var in tf.global_variables()
                    if var.name.split(':')[0] in saved_shapes])
  restore_vars = []
  name2var = dict(zip(map(lambda x:x.name.split(':')[0],
                          tf.global_variables()), tf.global_variables()))
  with tf.variable_scope('', reuse=True):
    for var_name, saved_var_name in var_names:
      curr_var = name2var[saved_var_name]
      var_shape = curr_var.get_shape().as_list()
      if var_shape == saved_shapes[saved_var_name]:
        restore_vars.append(curr_var)
  saver = tf.train.Saver(restore_vars)
  saver.restore(sess, save_file)


def train():
  with tf.Graph().as_default() as g:
    global_step = tf.contrib.framework.get_or_create_global_step()

    # ==========================================================================
    # Graph for main model
    # ==========================================================================
    x = tf.placeholder(tf.float32,
                       shape=[None, IMG_SIZE, IMG_SIZE, NUM_CHANNELS],
                       name='x')
    y = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES], name='y')
    phase = tf.placeholder(tf.bool, shape=[], name='phase')

    labels = tf.argmax(y, dimension=1)
  
    # Inference model.
    embeddings, logits = model.inference(x, phase,
        is_for_adver=tf.constant(False))
  
    # Calculate loss.
    loss, similarity_loss = model.loss(logits, labels, embeddings)
    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    if print_tensors:
      print ('Trainable variables')
      for var in tf.trainable_variables():
        print (var)
      print ('Loss')
      for loss in tf.get_collection('losses'):
        print (loss)
  
    # Calculate predictions.
    y_pred_cls = tf.argmax(logits, dimension=1)
    top_k = tf.nn.in_top_k(logits, labels, 1)
    true_count = tf.reduce_sum(tf.to_int32(top_k))
  
    # build train op.
    train_op, lr = model.train(total_loss, global_step)
    summary_op = tf.summary.merge_all()

    # ==========================================================================
    # Graph for adversarial training
    # ==========================================================================
    x_adver = tf.placeholder(tf.float32,
                             shape=[None, IMG_SIZE, IMG_SIZE, NUM_CHANNELS],
                             name='x_adver')
    y_adver = tf.placeholder(tf.float32,
                             shape=[None, NUM_CLASSES],
                             name='y_adver')

    # True for least likely, False for FGSM
    option_adver = tf.placeholder(tf.bool, shape=[], name='option_adver')
    ens_idx = tf.placeholder(tf.int32, shape=[], name='ens_idx')

    if FLAGS.ensemble:
      model_dict = {'per_image_standard': FLAGS.per_image_standard,
                    'add_noise_inference': FLAGS.add_noise_inference,
                    'model_name': FLAGS.model_name,
                    'resnet_n': FLAGS.resnet_n,
                    'embedding_at': FLAGS.embedding_at}

      main_scope_names = ['main', 'r20_ens_source', 'r110_ens_source']
      reuses = [True, False, False]
      resnet_ns = [FLAGS.resnet_n, 3, 18]
      # step forward
      grad_x_advers = [None] * 3
      ll_labels_list = [None] * 3
      for i, (reuse, main_scope_name, resnet_n) in \
          enumerate(zip(reuses, main_scope_names, resnet_ns)):
        model_dict['resnet_n'] = resnet_n
        logits_adver = model.step_forward(x_adver, model_dict,
                                          reuse, main_scope_name)
        ll_labels_list[i] = tf.argmin(logits_adver, dimension=1)
        adver_labels = tf.argmax(y_adver, dimension=1)
        # least likely labels or true labels
        feed_labels = tf.cond(option_adver,
                              lambda: ll_labels_list[i], lambda: adver_labels)
        # step backward
        grad_x_advers[i] = model.step_backward(x_adver, logits_adver,
                                               feed_labels, main_scope_name)
      grad_x_adver = tf.stack(grad_x_advers)[ens_idx]
      ll_labels = tf.stack(ll_labels_list)[ens_idx]
    else:
      # step forward
      logits_adver = model.step_forward(x_adver)
      ll_labels = tf.argmin(logits_adver, dimension=1)
      adver_labels = tf.argmax(y_adver, dimension=1)
      # least likely labels or true labels
      feed_labels = tf.cond(option_adver,
                            lambda: ll_labels, lambda: adver_labels)
      # step backward
      grad_x_adver = model.step_backward(x_adver, logits_adver, feed_labels)

    # ==========================================================================
    # Training preparation
    # ==========================================================================
    placeholder_dict = {'x': x, 'y': y, 'phase': phase, 'x_adver': x_adver,
                        'y_adver': y_adver, 'option_adver': option_adver}
    if FLAGS.ensemble:
      placeholder_dict['ens_idx'] = ens_idx

    tensor_dict = {'ll_labels': ll_labels, 'grad_x_adver': grad_x_adver,
                   'y_pred_cls': y_pred_cls, 'embeddings': embeddings}

    saver = tf.train.Saver()
    num_train_examples = len(data.train.images)
    max_steps = int(FLAGS.max_epochs * int(math.ceil(num_train_examples
                                                     /FLAGS.batch_size)))
    print ('max_epochs: %d, batch_size: %d, max_steps: %d' %
           (FLAGS.max_epochs, FLAGS.batch_size, max_steps))

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True

    # ==========================================================================
    # Tensorflow session creation
    # ==========================================================================
    with tf.Session(config=run_config) as sess:
      try:
        tf.global_variables_initializer().run()
      except:
        tf.initialize_all_variables().run()

      summary_writer = tf.summary.FileWriter(FLAGS.train_dir, g)
      start_step = 0

      if FLAGS.ensemble:
        ckpt = tf.train.get_checkpoint_state('./checkpoint/r20_ens_source')
        optimistic_restore(sess, ckpt.model_checkpoint_path)
        ckpt = tf.train.get_checkpoint_state('./checkpoint/r110_ens_source')
        optimistic_restore(sess, ckpt.model_checkpoint_path)

      if FLAGS.restore or FLAGS.restore_inplace:
        if FLAGS.restore_inplace:
          ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        else:
          ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
          start_step = int(
              ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
          print('resumed at global step: %d' % start_step)
          # Restores from checkpoint
          optimistic_restore(sess, ckpt.model_checkpoint_path)
        else:
          print('No checkpoint file found')

      # ========================================================================
      # Training
      # ========================================================================
      if FLAGS.is_train:
        iter = start_step
        for iter in xrange(start_step, max_steps):
          _start_time = time.time()
          x_batch, y_batch = get_batch(sess, placeholder_dict, tensor_dict)
  
          duration_batch = time.time() - _start_time
          feed_dict_train = {x: x_batch,
                             y: y_batch,
                             phase: True}
  
          if (iter+1) % 100 == 0:
            summary, _ = sess.run([summary_op, train_op],
                                  feed_dict=feed_dict_train)
            duration = time.time() - _start_time
            examples_per_sec = FLAGS.batch_size / duration
            sec_per_batch = float(duration)
            sec_per_get_batch = float(duration_batch)
            summary_writer.add_summary(summary, iter+1)
  
            global_step_val, lr_val, true_count_val, \
            loss_value, similarity_loss_val = sess.run(
                [global_step, lr, true_count, total_loss, similarity_loss],
                feed_dict=feed_dict_train)
  
            format_str = ('%s: step %d, lr = %.1e, acc = %.2f '
                          'loss = %.5f, similarity_loss = %.2e '
                          '(%.1f examples/sec; %.3f sec/batch; '
                          '%.3f sec/get_batch)')
            print (format_str % (datetime.now(), global_step_val, lr_val,
                                 float(true_count_val/FLAGS.batch_size),
                                 loss_value, similarity_loss_val,
                                 examples_per_sec, sec_per_batch,
                                 sec_per_get_batch))
          else:
             _ = sess.run([train_op], feed_dict=feed_dict_train)
  
          if (iter+1) % 500 == 0:
            print_test_accuracy(
                sess=sess,
                placeholder_dict=placeholder_dict,
                tensor_dict=tensor_dict,
                data_from='validation')
  
          if (iter+1) % 2000 == 0:
            save(sess, saver, FLAGS.train_dir, iter+1)
  
        if iter+1 == max_steps:
          save(sess, saver, FLAGS.train_dir, iter+1)

      # ========================================================================
      # Final evaluation
      # ========================================================================
      for max_e in test_e:
        for method in methods:
          for sparsity in sparsities:
            print ('%s test accuracy for e: %d, sparsity: %d'
                   % (method, max_e, sparsity))
            print_test_accuracy(
                sess=sess,
                placeholder_dict=placeholder_dict,
                tensor_dict=tensor_dict,
                data_from=test_data_from,
                max_e=max_e,
                method=method,
                sparsity=sparsity,
                show_example_errors=show_flag,
                show_confusion_matrix=show_cm_flag,
                print_accuracy=print_accuracy,
                save_adver_images=save_adver_images,
                use_saved_images=FLAGS.use_saved_images)
            if max_e == 0:
              break
          if max_e == 0:
            break

      # ========================================================================
      # Additional analysis
      # ========================================================================
      # Gradient correlation analysis
      if analyze_corr_grads_flag:
        test_idx = np.random.choice(10000, 128)
        analyze_corr_grads(
            sess=sess,
            placeholder_dict=placeholder_dict,
            tensor_dict=tensor_dict,
            data_from=test_data_from,
            test_idx=test_idx,
            sparsity=100,
            show_flag=show_analysis_flag)

      # Visualize logits (we used logits as embeddings)
      if visualize_embeddings_flag:
        test_idx = np.random.choice(10000, 128)
        visualize_embeddings(
            sess=sess,
            placeholder_dict=placeholder_dict,
            tensor_dict=tensor_dict,
            data_from=test_data_from,
            test_idx=test_idx,
            sparsity=100,
            show_flag=show_analysis_flag)

      # Plot trajectories of adversarial images with increased epsilon
      if trace_embeddings_flag:
        num_test_per_label = 1
        num_test = NUM_CLASSES * num_test_per_label
        test_idx = np.zeros(num_test, dtype=int)
        target_data = dataset_dict[test_data_from]
        for i in range(NUM_CLASSES):
          classes = np.argmax(target_data.labels, axis=1)
          test_idx[num_test_per_label*i:num_test_per_label*(i+1)] = \
               np.where(classes == i)[0][0:num_test_per_label]
        print(test_idx)
        trace_embeddings(
            sess=sess,
            placeholder_dict=placeholder_dict,
            tensor_dict=tensor_dict,
            data_from=test_data_from,
            test_idx=test_idx,
            sparsity=100,
            show_flag=show_analysis_flag)

      
def main(argv=None):
  pp.pprint(flags.FLAGS.__flags)
  restore = FLAGS.restore or FLAGS.restore_inplace
  if FLAGS.restore_inplace:
    checkpoint_dir = FLAGS.train_dir
  else:
    checkpoint_dir = FLAGS.checkpoint_dir
  if FLAGS.is_train:
    if tf.gfile.Exists(FLAGS.train_dir):
      if not (restore and FLAGS.train_dir == checkpoint_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
        tf.gfile.MakeDirs(FLAGS.train_dir)
    else:
      tf.gfile.MakeDirs(FLAGS.train_dir)
  train()

if __name__ == '__main__':
  tf.app.run()
