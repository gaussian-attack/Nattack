from provider import CIFAR10
import sys
import argparse
import tensorflow as tf
import numpy as np

from helpers import *
import sys
import os
import time
import pickle
import cv2

flags = tf.app.flags
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
flags.DEFINE_boolean("restore_inplace", True,
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
flags.DEFINE_string('data_dir', '../cifar10_data',
                    """Path to the CIFAR-10 data directory.""") ###################### change cifar10 dir
flags.DEFINE_string('eval_dir', '/tmp/mnist_eval',
                    """Directory where to write event logs.""")
flags.DEFINE_string('eval_data', 'test',
                    """Either 'test' or 'train_eval'.""")
flags.DEFINE_string('checkpoint_dir', '../../cascade_checkpoint/resnet110_cifar10_pivot_cascade',
                    """Directory where to read model checkpoints.""")
flags.DEFINE_string('train_dir', '/checkpoint/resnet110_cifar10_pivot',
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
flags.DEFINE_boolean("rand_crop", False,
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

from utils import *
import model

npop = 300     # population size
sigma = 0.1    # noise standard deviation
alpha = 0.008  # learning rate
# alpha = 0.001  # learning rate
boxmin = 0
boxmax = 1
boxplus = (boxmin + boxmax) / 2.
boxmul = (boxmax - boxmin) / 2.
folder = './liclipadvImages/'
epsi = 0.031

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--cifar-path', type=str, default='../cifar10_data/test_batch',
            help='path to the test_batch file from http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
    parser.add_argument('--perturb', type=str, default='perturb')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='./checkpoint/resnet110_cifar10_pivot_cascade/',
                        help='path to checkpoint file')
    args = parser.parse_args()

    test_loss = 0
    correct = 0
    total = 0
    faillist = []



    # set up TensorFlow session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    phase = tf.convert_to_tensor(False, dtype=tf.bool)
    input_xs = tf.placeholder(tf.float32, [None, 24, 24, 3])
    _, real_logits = model.inference(input_xs, phase,
                                     is_for_adver=tf.constant(False))

    # initialize a model

    ckpt_path = tf.train.get_checkpoint_state(args.checkpoint_dir)
    optimistic_restore(sess, ckpt_path.model_checkpoint_path)

    # initialize an attack (it's a white box attack, and it's allowed to look
    # at the internals of the model in any way it wants)
    # attack = BPDA(sess, model, epsilon=model.threat_model.epsilon, debug=args.debug)
    # attack = Attack(sess, model.model, epsilon=model.threat_model.epsilon)

    # initialize a data provider for CIFAR-10 images
    provider = CIFAR10(FLAGS.data_dir + '/test_batch')
    start = 0
    end = 10000
    total = 0
    successlist = []
    printlist = []

    start_time = time.time()
    perturbs = os.listdir('./')
    all_dir = []
    for x in perturbs:
      if 'black' in x:
        all_dir.append(x)


    for y in all_dir:
      perturb_files = os.listdir(y)
      numbers = []
      totalImages = 0
      succImages = 0

      for x in perturb_files:
        number = x.split('_')[1]
        name = x.split('_')[0]
        number1 = int(number.split('.pkl')[0])
        numbers.append(number1)

      for i in numbers:
        success = False
        #print('evaluating %d of [%d, %d)' % (i, start, end), file=sys.stderr)
        inputs, targets= provider[i]

        inputs = inputs.reshape((1, 32, 32, 3))
        inputs = copy_crop_images(inputs, 24, True)
        inputs = inputs[0][0]

        in_pkl = y+ '/'+ name + '_' + str(i)+'.pkl'
        ##### thermometer encoding

        logits = sess.run(real_logits,feed_dict={input_xs: [inputs]})
        if np.argmax(logits) != targets:
            print('skip the wrong example ', i)
            continue
        totalImages += 1
        modify = pickle.load(open(in_pkl, 'rb'))
        #print('modify : ',modify.shape)
        modify = cv2.resize(modify[0].transpose(1,2,0), dsize=(24,24), interpolation=cv2.INTER_LINEAR)
        modify = modify.transpose(2, 0, 1)
        modify = modify.reshape((1,3,24,24))
        newimg = torch_arctanh((inputs - boxplus) / boxmul).transpose(2, 0, 1)
        realinputimg = np.tanh(newimg + modify) * boxmul + boxplus
        realdist = realinputimg - (np.tanh(newimg) * boxmul + boxplus)
        realclipdist = np.clip(realdist, -epsi, epsi)
        realclipinput = realclipdist + (np.tanh(newimg) * boxmul + boxplus)
        l2real = np.sum((realclipinput - (np.tanh(newimg) * boxmul + boxplus)) ** 2) ** 0.5
        outputsreal = sess.run(real_logits, feed_dict={input_xs: realclipinput.transpose(0, 2, 3, 1)})
        if (np.argmax(outputsreal) != targets) and (np.abs(realclipdist).max() <= epsi):
            succImages += 1

      success_rate = succImages / float(totalImages)
      print('name : ', y)
      print('succ rate', success_rate)
      print('succ {} , total {}'.format(succImages, totalImages))


if __name__ == '__main__':
    main()
