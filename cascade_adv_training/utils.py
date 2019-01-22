# Part of this code comes from
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
# ==============================================================================

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import collections
import itertools
from dataset import DataSet

flags = tf.app.flags
FLAGS = flags.FLAGS
# flags.DEFINE_string('data_set', 'cifar10',
#                     """Choose between mnist, cifar10.""")
# FLAGS = flags


# We initially used 24x24 for size of cifar input images.
# Change this parameter if needed.
CIFAR_PAD_SIZE = 0
CIFAR_IMG_SIZE = 24

# These are the parameters for plot.
LINE_WIDTH = 4
FONT_SIZE = 16
LEGEND_FONT_SIZE = 20

# ==============================================================================
# Load the data
# ==============================================================================
if FLAGS.data_set == 'mnist':
  IMG_RAW_SIZE = 28
  IMG_SIZE = 28
  NUM_CHANNELS = 1
  IMG_RAW_SIZE_FLAT = IMG_RAW_SIZE^2 * NUM_CHANNELS
  IMG_SHAPE = (IMG_SIZE, IMG_SIZE)
  NUM_CLASSES = 10
  mean_image = np.zeros((IMG_SIZE, IMG_SIZE, NUM_CHANNELS))

  # Load MNIST data
  def data_mnist(img_size, num_channels, one_hot=True):
    """
    Preprocess MNIST dataset
    """
    from keras.datasets import mnist
    from keras.utils import np_utils
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0],
                              img_size,
                              img_size,
                              num_channels)

    X_test = X_test.reshape(X_test.shape[0],
                            img_size,
                            img_size,
                            num_channels)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    print("Loaded MNIST test data.")

    if one_hot:
      # convert class vectors to binary class matrices
      y_train = np_utils.to_categorical(y_train, 10).astype(np.float32)
      y_test = np_utils.to_categorical(y_test, 10).astype(np.float32)

    return X_train, y_train, X_test, y_test

  # Make dataset objects for MNIST
  x_train, t_train, x_test, t_test = data_mnist(IMG_SIZE, NUM_CHANNELS)

elif 'cifar' in FLAGS.data_set:
  from cifar_load import *
  if not os.path.exists(FLAGS.data_dir):
      raise ValueError('data dir not exist')
  if FLAGS.data_set == 'cifar10':
    x_train, t_train, x_test, t_test = cifar10(FLAGS.data_dir, dtype='float32')
    NUM_CLASSES = 10
  elif FLAGS.data_set == 'cifar100':
    x_train, t_train, x_test, t_test = cifar100(FLAGS.data_dir, dtype='float32')
    NUM_CLASSES = 100
  else:
    raise ValueError('Please choose between mnist, cifar10 and cifar100')
  PAD_SIZE = CIFAR_PAD_SIZE
  IMG_SIZE = CIFAR_IMG_SIZE

  NUM_CHANNELS = 3
  IMG_RAW_SIZE = 32 + 2*PAD_SIZE
  IMG_RAW_SIZE_FLAT = IMG_RAW_SIZE^2 * NUM_CHANNELS
  IMG_SIZE_FLAT = IMG_SIZE * IMG_SIZE * NUM_CHANNELS
  IMG_SHAPE = (IMG_SIZE, IMG_SIZE)

  if FLAGS.norm_option == 0: # per pixel normalization
    mean_image = x_train.mean(axis=(0))
    print ('per pixel normalization.')
  elif FLAGS.norm_option == 1: # per channel normalization
    mean_image = x_train.mean(axis=(0,1,2))
    print ('per channel normalization.')
  elif FLAGS.norm_option == 2: # per channel normalization
    mean_image = 0.
    print ('no normalization.')
  else:
    raise ValueError('Please choose between 0, 1 and 2')

  x_train = x_train - mean_image
  x_test = x_test - mean_image

  # Normalize mean_image since the data will be normalized
  # We will add mean_image to the sample images when plotting
  mean_image = mean_image/255.

  # Crop mean_image if necessary.
  if FLAGS.norm_option == 0:
    def center_crop_image(image, img_size):
      img_raw_size = image.shape[0]
      if img_raw_size > img_size:
          offset = int((img_raw_size-img_size)/2)
          result = image[offset:offset+img_size,offset:offset+img_size,:]
      elif img_raw_size == img_size:
        result = image
      else:
        raise ValueError('target image size must be equal '
                          'or less than original image size.')
      return result

    mean_image = center_crop_image(mean_image, IMG_SIZE)

  # Zero padding if necessary.
  def pad_images(img_batch, pad_size=0, org_batch=None):
    if pad_size > 0:
      batch_size = img_batch.shape[0]
      img_size = img_batch.shape[1]
      num_channels = img_batch.shape[3]
      if org_batch is None:
        result = np.zeros([batch_size, img_size+2*pad_size,
                           img_size+2*pad_size, num_channels])
      else:
        result = np.copy(org_batch)
      result[:, pad_size:pad_size+img_size, \
             pad_size:pad_size+img_size,:] = img_batch
    elif pad_size == 0:
      result = img_batch
    else:
      raise ValueError('pad_size must be a positive integer.')
    return result

  x_train = pad_images(x_train, PAD_SIZE)
  x_test = pad_images(x_test, PAD_SIZE)

  # Prepare for cascade adversarial training.
  # We anticipate the pre-defined naming convention.
  if FLAGS.cascade:
    iter_fgsm_e = []
    for e in range(1, FLAGS.max_e+1, 1):
      iter_fgsm_e.append(e)
    iter_fgsm_datasets = []
    for max_e in iter_fgsm_e:
      filename = FLAGS.saved_iter_fgsm_dir + '/train_iter_fgsm_' + str(max_e) \
                 + '_' + str(100) + '.npy'
      iter_fgsm_image = np.load(filename)
      iter_fgsm_image = pad_images(iter_fgsm_image,
                                   (32 - CIFAR_IMG_SIZE)/2+PAD_SIZE, x_train)
      iter_fgsm_datasets.append(
          DataSet(iter_fgsm_image, t_train, reshape=False))

else:
  raise ValueError('Please choose between mnist, cifar10 and cifar100')

# Make dataset objects.
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
train = DataSet(x_train, t_train, reshape=False)
validation = DataSet(x_test, t_test, reshape=False)
test = DataSet(x_test, t_test, reshape=False)
data = Datasets(train=train, validation=validation, test=test)
del x_train, t_train, x_test, t_test

dataset_dict = {'validation': data.validation,
                'test': data.test,
                'train': data.train}


print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))

# ==============================================================================
# Helper function
# ==============================================================================
def copy_crop_images(img_batch, img_size,
                     rand_crop=False,
                     start_xy=None,
                     flip_flag=None):
  """Returns copied version of images after crop and flip.
     If this function is called

  Args:
    img_batch: a numpy array of images to be copied.
    img_size: width/height of desired images.
    rand_crop: True for random crop.
    start_xy: if start_xy is not None, we use start_xy as a offset
              for cropping. shape: [batch_size, 2]
    flip_flag: if start_xy is not None, flip_flag should not be None.
               And we use use flip_flag to determine whether to flip or not
               to flip the images.
               start_xy and flip_flag can be used when we combine cascade
               adversarial training and similarity learning to ensure
               two clean and adversarial images are well aligned.
               Initially we added this feature to enable random crop for
               iter_fgsm images, however, we decided to perform center crop
               for iter_fgsm images to maximize the effect of adversarial
               noises.
  Returns:
    result: copied version of images after crop and flip
    start_xy: a numpy array of offset used for cropping. This output can be
              used as an input to this function when it is called again.
    flip_flag: a numpy array which indicates whether the output is flipped
               or not. This output can be also used as an input to the same
               function when it is called again.
  """
  img_raw_size = img_batch.shape[1]
  batch_size = img_batch.shape[0]
  num_channels = img_batch.shape[3]
  
  result = np.empty([batch_size, img_size, img_size, num_channels])
  if img_raw_size > img_size:
    if rand_crop:
      diff = img_raw_size - img_size
      if start_xy is None:
        start_xy = np.random.randint(0, diff+1, size=[batch_size, 2])
        flip_flag = np.random.randint(0, 2, size=batch_size)
      end_xy = start_xy + img_size
      for i in range(batch_size):
        cropped_image = img_batch[i,start_xy[i,0]:end_xy[i,0],
                                  start_xy[i,1]:end_xy[i,1], :]
        if flip_flag[i] == 1:
          cropped_image = cropped_image[:,::-1,:]
        result[i] = cropped_image
    else:
      offset = int((img_raw_size-img_size)/2)
      result = np.copy(img_batch[:,offset:offset+img_size,
                                 offset:offset+img_size,:])
  elif img_raw_size == img_size:
    result = np.copy(img_batch)
  else:
    raise ValueError('target image size must be equal or less than '
                     'original image size.')

  return result, start_xy, flip_flag

def get_intensity(size, max_e, adver_option=0):
  """Returns intensities which will be used for selecting epsilon for
     adversarial images.

  Args:
    size: batch_size
    max_e: maximum of epsilon for adversarial perturbation
    adver_option: option for intensity distribution
                  0: max_e is used for all adversarial images
                  1: uniform[1, max_e]
                  2: truncated normal distribution ~ abs(N(0, max_e/2))
                  3: truncated normal distribution shifted with x=max_e
  Returns:
    intensity: a numpy array of intensities which will be used for
               selecting epsilon for adversarial images.
  """


  if adver_option == 0:
    intensity = max_e*np.ones(size)
  elif adver_option == 1:
    intensity = np.random.randint(1, max_e+1, size=size)
  elif adver_option == 2:
    intensity = np.clip(abs(np.random.normal(0, max_e/2, size=size)),
                        0, max_e)
  elif adver_option == 3:
    intensity = -np.clip(abs(np.random.normal(0, max_e/2, size=size)),
                         0, max_e) + max_e
  else:
    raise ValueError('Choose option 0: max_e, 1: uniform[1, max_e], '
                     '2: truncated normal, 3: y-axis shifted truncated normal')

  return intensity

def get_step_adver_image(images, adver_noise, max_e, adver_option=0,
                         per_pixel_rand=False, sparsity=100):
  """Returns adversarial images and intensities of them.

  Args:
    images: a numpy array of images used for adversarial images generation
    adver_noise: adversarial noises generated from the below methods.
                 'step_ll': one-step least likely method
                 'step_fgsm': one-step fast gradient sign method (FGSM)
                 'step_rand': one-step random sign method
                 'iter_ll': iterative least likely method
                 'iter_fgsm': iterative fast gradient sign method
    max_e: maximum of epsilon for adversarial perturbation
    adver_option: option for intensity distribution
                  0: max_e is used for all adversarial images
                  1: uniform[1, max_e]
                  2: truncated normal distribution ~ abs(N(0, max_e/2))
                  3: truncated normal distribution shifted with x=max_e
    per_pixel_rand: True to add additional per pixel normal distribution
                     ~ N(0, 1). Default: False
    sparsity: value in range [0, 100] to control sparsity of perturbation.
              If sparsity < 100, only sparsity (%) of pixels have non-zero
              adversarial noises.

  Returns:
    result: a numpy array of adversarial images
    intensity: a numpy array of epsilon values of adversarial images
  """

  intensity = get_intensity(len(adver_noise), max_e, adver_option)

  if per_pixel_rand:
    e = np.clip(abs(intensity[:,None,None,None]
                    + np.random.normal(0, 1, size=adver_noise.shape)),
                    0, max_e)/255.
  else:
    e = intensity[:,None,None,None]/255.

  # if sparsity == 100, then just simple sign method
  # sparsity < 100, means , non-zero values are sparsity (%)
  if sparsity == 100:
    mask = np.ones_like(adver_noise)
  else:
    threshold = np.percentile(abs(adver_noise), 100-sparsity, axis=(1,2,3))
    idx = np.where(abs(adver_noise) >= threshold[:,None,None,None])
    mask = np.zeros_like(adver_noise)
    mask[idx] = 1.
  adver_noise = e*mask*np.sign(adver_noise)

  intensity = intensity/float(FLAGS.max_e)

  # make sure the mean added values are in the valid range [0, 1]
  images = np.clip(images + adver_noise + mean_image, 0., 1.)
  result = images - mean_image
  return result, intensity

def get_adver_image(sess, placeholder_dict, tensor_dict, images, labels, max_e,
                    method='step_ll', adver_option=0, per_pixel_rand=False,
                    sparsity=100, rand_seed=None):
  """Returns adversarial images and intensities

  Args:
    sess: tensorflow session
    placeholder_dict: dictionary of tensorflow placeholders
    tensor_dict: dictionary of tensorflow nodes
    images: input images for feed
    labels: input labels for feed
    max_e: maximum of epsilon for adversarial perturbation
    method: method of adversarial images generation. Posible values are
            'step_ll': one-step least likely method
            'step_fgsm': one-step fast gradient sign method (FGSM)
            'step_rand': one-step random sign method
            'iter_ll': iterative least likely method
            'iter_fgsm': iterative fast gradient sign method
    adver_option: option for intensity distribution
                  0: max_e is used for all adversarial images
                  1: uniform[1, max_e]
                  2: truncated normal distribution ~ abs(N(0, max_e/2))
                  3: truncated normal distribution shifted with x=max_e
    per_pixel_rand: True to add additional per pixel normal distribution
                     ~ N(0, 1). Default: False
    sparsity: value in range [0, 100] to control sparsity of perturbation.
              If sparsity < 100, only sparsity (%) of pixels have non-zero
              adversarial noises.
    rand_seed: random seed for 'step_rand' method


  Returns:
    new_batch: mini-batch for input images (numpy array)
    new_y_batch: mini-batch for one-hot encoded true labels (numpy array)
  """

  feed_dict_train = {}
  if 'ens_idx' in placeholder_dict:
    ens_idx = placeholder_dict['ens_idx']
    feed_dict_train[ens_idx] = np.random.randint(3)

  x_adver = placeholder_dict['x_adver']
  y_adver = placeholder_dict['y_adver']
  option_adver = placeholder_dict['option_adver']

  ll_labels = tensor_dict['ll_labels']
  grad_x_adver = tensor_dict['grad_x_adver']

  if 'll' in method:
    feed_dict_train.update({x_adver: images, y_adver: labels,
                            option_adver: True})
    # we first get the target least likely labels as well as gradients
    # we subtract the gradient from the images since we want to
    # minimize the loss for the wrong target labels.
    target_labels, adver_noise = sess.run([ll_labels, grad_x_adver],
                                          feed_dict=feed_dict_train)
    adver_noise = -adver_noise
    target_ll_labels = np.zeros_like(labels)
    target_ll_labels[np.arange(len(target_ll_labels)), target_labels] = 1

  elif 'fgsm' in method:
    feed_dict_train.update({x_adver: images, y_adver: labels,
                            option_adver: False})
    adver_noise = sess.run(grad_x_adver, feed_dict=feed_dict_train)
  elif 'both' in method:
    adver_batch_size = len(images)
    # first half : least likely
    feed_dict_train.update({x_adver: images[:adver_batch_size/2],
                            y_adver: labels[:adver_batch_size/2],
                            option_adver: True})
    target_labels, adver_noise_ll = sess.run([ll_labels, grad_x_adver],
                                             feed_dict=feed_dict_train)
    adver_noise_ll = -adver_noise_ll

    # second half : fgsm
    feed_dict_train.update({x_adver: images[adver_batch_size/2:],
                            y_adver: labels[adver_batch_size/2:],
                            option_adver: False})
    adver_noise_fgsm = sess.run(grad_x_adver, feed_dict=feed_dict_train)
    adver_noise = np.concatenate([adver_noise_ll, adver_noise_fgsm], axis=0)

  elif 'rand' in method:
    np.random.seed(rand_seed)
    adver_noise = np.random.normal(0, 1, size=images.shape)
  else:
    raise ValueError('method should be one of step_ll, step_fgsm, iter_ll, '
                     'and iter_fgsm.')

  if 'step' in method:
    result, intensity = get_step_adver_image(images, adver_noise, max_e,
        adver_option, per_pixel_rand, sparsity)
  elif 'iter' in method:
    org_images = np.copy(images)
    max_iter = int(np.floor(min(4+max_e, 1.25*max_e))) -1
    images, _ = get_step_adver_image(images, adver_noise, 1, adver_option,
                                     per_pixel_rand, sparsity)
    for i in range(max_iter):
      if 'll' in method:
        feed_dict_train = {x_adver: images, y_adver: target_ll_labels,
                           option_adver: False}
        adver_noise = -sess.run(grad_x_adver, feed_dict=feed_dict_train)
      elif 'fgsm' in method:
        feed_dict_train = {x_adver: images, y_adver: labels,
                           option_adver: False}
        adver_noise = sess.run(grad_x_adver, feed_dict=feed_dict_train)

      images, _ = get_step_adver_image(images, adver_noise, 1, adver_option,
                                       per_pixel_rand, sparsity)
      if i > max_e-2:
        diff = images - org_images
        diff = np.clip(diff, -max_e/255., max_e/255.)
        images = org_images + diff

    result = images
    intensity = np.ones(len(images))*max_e/float(FLAGS.max_e)
  else:
    raise ValueError('method should be one of step_ll, step_fgsm, iter_ll, '
                     'and iter_fgsm.')

  return result, intensity

def get_batch(sess, placeholder_dict, tensor_dict):
  """Returns mini-batch for training based on FLAGS setting

  Args:
    sess: tensorflow session
    placeholder_dict: dictionary of tensorflow placeholders
    tensor_dict: dictionary of tensorflow nodes

  Returns:
    new_batch: mini-batch for input images (numpy array)
    new_y_batch: mini-batch for one-hot encoded true labels (numpy array)
  """

  batch_size = FLAGS.batch_size
  adver_batch_size = FLAGS.adver_batch_size
  max_e = FLAGS.max_e
  adver_option = FLAGS.adver_option
  per_pixel_rand = FLAGS.per_pixel_rand
  sparsity = FLAGS.sparsity
  method = FLAGS.method

  # Fetch the initial batch
  if FLAGS.similarity_loss_factor > 0.0 or FLAGS.pivot_loss_factor > 0.0:
    assert adver_batch_size > 0
    fetch_batch_size = max(adver_batch_size, batch_size-adver_batch_size)
    x_raw_batch, y_batch, perm = data.train.next_batch(fetch_batch_size)
  else:
    x_raw_batch, y_batch, perm = data.train.next_batch(batch_size)
  x_batch, start_xy, flip_flag = copy_crop_images(
      x_raw_batch, IMG_SIZE, FLAGS.rand_crop)

  # For cascade adversarial training, prepare list of iter_fgsm images
  # Currently, only supported for CIFAR
  if FLAGS.cascade:
    if FLAGS.similarity_loss_factor > 0.0 or FLAGS.pivot_loss_factor > 0.0:
      iter_fgsm_batch_size = fetch_batch_size
      # For similarity learning, we substitute center crop for random crop
      # since this will be compared with iter_fgsm dataset
      # which is center cropped.
      x_batch_center_crop, s_temp, f_temp = copy_crop_images(
          x_raw_batch[adver_batch_size/2:adver_batch_size],
          IMG_SIZE, rand_crop=False)
      x_batch[adver_batch_size/2:adver_batch_size] = x_batch_center_crop
    else:
      iter_fgsm_batch_size = batch_size

    x_iter_fgsm_batches = []
    for iter_fgsm_dataset in iter_fgsm_datasets:
      x_iter_fgsm_batch, y_temp, perm_temp = iter_fgsm_dataset.next_batch(
          iter_fgsm_batch_size, perm=perm)
      x_iter_fgsm_batch, s_temp, f_temp = copy_crop_images(x_iter_fgsm_batch,
          IMG_SIZE, False, start_xy, flip_flag)
      x_iter_fgsm_batches.append(x_iter_fgsm_batch)

  # Prepare new mini batch
  new_batch = np.empty([batch_size, x_batch.shape[1],
                       x_batch.shape[2], x_batch.shape[3]])
  new_y_batch = np.empty([batch_size, y_batch.shape[1]])

  # Prepare adversarial example patch
  if FLAGS.adversarial:
    assert adver_batch_size > 0
    if FLAGS.cascade:
      adver_images = np.empty_like(x_batch[0:adver_batch_size])
      intensity = np.zeros([adver_batch_size])
      images = x_batch[0:adver_batch_size/2]
      labels = y_batch[0:adver_batch_size/2]
      # half of the adversarial images from the fast method
      adver_images[0:adver_batch_size/2], \
      intensity[0:adver_batch_size/2] =  get_adver_image(
          sess=sess,
          placeholder_dict=placeholder_dict,
          tensor_dict=tensor_dict,
          images=images,
          labels=labels,
          max_e=max_e,
          method=method,
          adver_option=adver_option,
          per_pixel_rand=per_pixel_rand,
          sparsity=sparsity)
      # second half of the adversarial images from the saved iter_fgsm method
      intensity_temp = get_intensity(adver_batch_size/2, max_e, adver_option)
      intensity_idx = np.clip(np.floor(intensity_temp), 0, max_e-1)
      intensity[adver_batch_size/2:] = intensity_idx + 1.
      for i, idx in enumerate(intensity_idx):
        adver_images[adver_batch_size/2+i] = \
            x_iter_fgsm_batches[int(idx)][adver_batch_size/2+i]
    else:
      images = x_batch[0:adver_batch_size]
      labels = y_batch[0:adver_batch_size]
      adver_images, intensity =  get_adver_image(
          sess=sess,
          placeholder_dict=placeholder_dict,
          tensor_dict=tensor_dict,
          images=images,
          labels=labels,
          max_e=max_e,
          method=method,
          adver_option=adver_option,
          per_pixel_rand=per_pixel_rand,
          sparsity=sparsity)

    new_batch[0:adver_batch_size] = adver_images
    new_y_batch[0:adver_batch_size] = y_batch[0:adver_batch_size]

    # If similarity loss is applied, assign the corresponding clean images
    # to the rest of the mini batch.
    # If mini batch size = m, adver_batch_size = k,
    # first k images are adversarial images, m-k images are the clean images.
    if FLAGS.similarity_loss_factor > 0.0 or FLAGS.pivot_loss_factor > 0.0:
      new_batch[adver_batch_size:] = x_batch[0:batch_size-adver_batch_size]
      new_y_batch[adver_batch_size:] = y_batch[0:batch_size-adver_batch_size]
    else:
      new_batch[adver_batch_size:] = x_batch[adver_batch_size:]
      new_y_batch[adver_batch_size:] = y_batch[adver_batch_size:]
  else:
    new_batch = x_batch
    new_y_batch = y_batch
  del x_batch, y_batch
  return new_batch, new_y_batch

def get_data(sess=None, placeholder_dict=None, tensor_dict=None,
             data_from='validation', custom_data=None, custom_labels=None,
             batch_size=1, max_e=0, method='step_ll', sparsity=100,
             rand_seed=None):
  """Returns adversarial images dataset.
     If custom_data is not None, we use custom_data instead of global dataset.
     If max_e == 0, then, clean images are returned.

  Args:
    sess: tensorflow session
    placeholder_dict: dictionary of tensorflow placeholders
    tensor_dict: dictionary of tensorflow nodes
    data_from: string to indicate target dataset
               'training': training dataset
               'validation': validation dataset
               'test': test dataset
    custom_data: if custom_data is not None, we use this as the input images
                 for adversarial images generation instead of images from
                 the dataset.
    custom_labels: if custom_labels is not None, we use this as the input labels
                   for adversarial images generation instead of labels from
                   the dataset.
    batch_size: mini batch size for session run
    max_e: maximum of epsilon for adversarial perturbation
    method: method of adversarial images generation. Posible values are
            'step_ll': one-step least likely method
            'step_fgsm': one-step fast gradient sign method (FGSM)
            'step_rand': one-step random sign method
            'iter_ll': iterative least likely method
            'iter_fgsm': iterative fast gradient sign method
    sparsity: value in range [0, 100] to control sparsity of perturbation.
              If sparsity < 100, only sparsity (%) of pixels have non-zero
              adversarial noises.
    rand_seed: random seed for 'step_rand' method


  Returns:
    new_batch: mini-batch for input images (numpy array)
    new_y_batch: mini-batch for one-hot encoded true labels (numpy array)
  """

  target_data = dataset_dict[data_from]

  if custom_data is not None:
    target_images = np.copy(custom_data)
    num_test = len(target_images)
    target_labels = custom_labels
  else:
    target_images, start_xy, flip_flag = copy_crop_images(target_data.images,
                                                          IMG_SIZE)
    num_test = len(target_images)
    target_labels = target_data.labels[0:num_test]

  # make adversarial images if necessary
  if max_e != 0:
    # The starting index for the next batch is denoted i.
    i = 0
    while i < num_test:
      # The ending index for the next batch is denoted j.
      if num_test-i < batch_size:
        j = num_test
      else:
        j = i + batch_size

      # Get the images from the test-set between index i and j.
      images = target_images[i:j, :]
      labels = target_labels[i:j, :]
      images, intensities =  get_adver_image(
          sess=sess,
          placeholder_dict=placeholder_dict,
          tensor_dict=tensor_dict,
          images=images,
          labels=labels,
          max_e=max_e,
          method=method,
          adver_option=0,
          per_pixel_rand=False,
          sparsity=sparsity,
          rand_seed=rand_seed)

      target_images[i:j, :] = images

      # Set the start-index for the next batch to the
      # end-index of the current batch.
      i = j
  return target_images, target_labels

def plot_images(images, cls_true, cls_pred=None):
  """Plot examples.

  Args:
    images: a numpy array of images.
            We expect the length of images is 9.
    cls_true: a numpy array of true classes for the test images.
    cls_pred: a numpy array of predictions for the test images.
  """
  # Assume normalized images for color images like cifar
  assert len(images) == len(cls_true) == 9

  # Create figure with 3x3 sub-plots.
  fig, axes = plt.subplots(3, 3)
  fig.subplots_adjust(hspace=0.3, wspace=0.3)

  for i, ax in enumerate(axes.flat):
    # Get the i'th image and reshape the array.
    image = images[i]

    # Ensure the noisy pixel-values are between 0 and 1.
    if NUM_CHANNELS == 3:
      image = image + mean_image
    image = np.clip(image, 0.0, 1.0)

    # Plot image.
    if NUM_CHANNELS == 1:
      image = image.reshape(IMG_SHAPE)
      ax.imshow(image, cmap='binary', interpolation='nearest')
    else:
      ax.imshow(image, interpolation='nearest')

    # Show true and predicted classes.
    if cls_pred is None:
      xlabel = "True: {0}".format(cls_true[i])
    else:
      xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

    # Show the classes as the label on the x-axis.
    ax.set_xlabel(xlabel)

    # Remove ticks from the plot.
    ax.set_xticks([])
    ax.set_yticks([])

  # Ensure the plot is shown correctly with multiple plots
  # in a single Notebook cell.
  plt.show(block=False)


def plot_example_errors(test_images, cls_true, cls_pred, correct):
  """Plot examples.

  Args:
    test_images: a numpy array of test images.
    cls_true: a numpy array of true classes for the test images.
    cls_pred: a numpy array of predictions for the test images.
    correct: a boolean array whether the predicted class is
             equal to the true class for each image.
  """
  # Negate the boolean array.
  incorrect = (correct == False)

  images = test_images[incorrect]

  # Get the predicted classes for those images.
  cls_pred = cls_pred[incorrect]

  # Get the true classes for those images.
  cls_true = cls_true[incorrect]

  # Get the adversarial noise from inside the TensorFlow graph.
  # Plot the first 9 images.
  plot_images(images=images[0:9],
        cls_true=cls_true[0:9],
        cls_pred=cls_pred[0:9])


def plot_confusion_matrix(cls_true, cls_pred):
  """Print confusion matrix.
     Get the confusion matrix using sklearn.

  Args:
    cls_true: a numpy array of true classes for the test images.
    cls_pred: a numpy array of predictions for the test images.
  """
  cm = confusion_matrix(y_true=cls_true,
             y_pred=cls_pred)
  # Print the confusion matrix as text.
  print(cm)
  print(np.sum(cm, axis=0))


def print_test_accuracy(sess, placeholder_dict, tensor_dict,
                        data_from='validation', max_e=0,
                        method='step_ll', sparsity=100,
                        show_example_errors=False, show_confusion_matrix=False,
                        print_accuracy=True, save_adver_images=False,
                        use_saved_images=False, transfer_analysis=False):
  """Print test accuracies.

  Args:
    sess: tensorflow session
    placeholder_dict: dictionary of tensorflow placeholders
    tensor_dict: dictionary of tensorflow nodes
    data_from: string to indicate target dataset
               'training': training dataset
               'validation': validation dataset
               'test': test dataset
    max_e: maximum of epsilon for adversarial perturbation
    method: method of adversarial images generation. Posible values are
            'step_ll': one-step least likely method
            'step_fgsm': one-step fast gradient sign method (FGSM)
            'step_rand': one-step random sign method
            'iter_ll': iterative least likely method
            'iter_fgsm': iterative fast gradient sign method
    sparsity: value in range [0, 100] to control sparsity of perturbation.
              If sparsity < 100, only sparsity (%) of pixels have non-zero
              adversarial noises.
    show_example_errors: True for showing example errors.
    show_confusion_matrix: True for showing confusion matrix.
    print_accuracy: True for printing accuracy.
    save_adver_images: True to save generated adversarial images.
                       The saved images will be used for cascade adversarial
                       training (if the images are generated from training set)
                       or used for transfer analysis (if the images are
                       generated from validation/test set).
    use_saved_images: True to test saved adversarial images generated from
                      other networks. The images should be stored before
                      with other networks.
    transfer_analysis: True to perform transfer rate analysis described in
                       (Kurakin et. al 2017.)
                       Paper link: https://arxiv.org/pdf/1611.01236.pdf

  """

  x = placeholder_dict['x']
  phase = placeholder_dict['phase']

  y_pred_cls = tensor_dict['y_pred_cls']

  batch_size = FLAGS.batch_size

  if use_saved_images and max_e > 0:
    filename = FLAGS.saved_data_dir + '/' + data_from + '_' + method + '_' \
               + str(max_e) + '_' + str(sparsity) + '.npy'
    test_images = np.load(filename)
    print ('Loading test images from %s' % filename)
    ref_clean_bool_filename = FLAGS.saved_data_dir + '/' + data_from \
                              + '_step_ll_0_100' + '_bool.npy'
    ref_bool_filename = FLAGS.saved_data_dir + '/' + data_from + '_' + method \
                        + '_' + str(max_e) + '_' + str(sparsity) + '_bool.npy'

    # Prepare for transfer rate analysis described in (Kurakin et al. 2017)
    # Paper link: https://arxiv.org/pdf/1611.01236.pdf
    if transfer_analysis:
      test_clean_bool_filename = FLAGS.train_dir + '/' + data_from \
                                 + '_step_ll_0_100' + '_bool.npy'
      ref_correct_clean = np.load(ref_clean_bool_filename)
      ref_correct = np.load(ref_bool_filename)
      test_correct_clean = np.load(test_clean_bool_filename)
      # index for clean classified correctly, adversarial misclassified
      clean_T_adver_F_idx = np.where(ref_correct_clean*(~ref_correct)==True)
      both_clean_T_adver_F_idx = np.where(test_correct_clean*
                                 ref_correct_clean*(~ref_correct)==True)

    test_data = dataset_dict[data_from]
    # Number of images in the test-set.
    num_test = len(test_images)
    test_labels = test_data.labels[0:num_test]
  # make adversarial images if necessary
  else:
    test_images, test_labels = get_data(
        sess=sess,
        placeholder_dict=placeholder_dict,
        tensor_dict=tensor_dict,
        data_from=data_from,
        batch_size=batch_size,
        max_e=max_e,
        method=method,
        sparsity=sparsity)

    if save_adver_images:
      filename = FLAGS.train_dir + '/' + data_from + '_' + method + '_' \
                 + str(max_e) + '_' + str(sparsity) + '.npy'
      np.save(filename, test_images)
      print ('Writing test images to %s' % filename)

  if not print_accuracy:
    return

  # Number of images in the test-set.
  num_test = len(test_images)

  # Allocate an array for the predicted classes which
  # will be calculated in batches and filled into this array.
  cls_pred = np.zeros(shape=num_test, dtype=np.int)

  # Now calculate the predicted classes for the batches.
  # We will just iterate through all the batches.
  # There might be a more clever and Pythonic way of doing this.

  # The starting index for the next batch is denoted i.
  i = 0
  correct_sum = 0
  while i < num_test:
    # The ending index for the next batch is denoted j.
    if num_test-i < FLAGS.batch_size:
      j = num_test
    else:
      j = i + FLAGS.batch_size

    # Get the images from the test-set between index i and j.
    images = test_images[i:j, :]

    # Create a feed-dict with these images
    feed_dict = {x: images, phase: False}

    # Calculate the predicted class using TensorFlow.
    cls_pred[i:j] = sess.run(y_pred_cls, feed_dict=feed_dict)

    # Set the start-index for the next batch to the
    # end-index of the current batch.
    i = j

  # Convenience variable for the true class-numbers of the test-set.
  cls_true = np.argmax(test_labels, axis=1)

  # Create a boolean array whether each image is correctly classified.
  correct = (cls_true == cls_pred)

  if save_adver_images:
    filename = FLAGS.train_dir + '/' + data_from + '_' + method + '_' \
               + str(max_e) + '_' + str(sparsity) + '_bool.npy'
    np.save(filename, correct)
    print ('Writing boolean array to %s' % filename)

  # Calculate the number of correctly classified images.
  # When summing a boolean array, False means 0 and True means 1.
  correct_sum = correct.sum()

  # Classification accuracy is the number of correctly classified
  # images divided by the total number of images in the test-set.
  acc = float(correct_sum) / num_test

  # Print the accuracy.
  msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
  print(msg.format(acc, correct_sum, num_test))

  # Print the transfer rate described in (Kurakin et. al 2017)
  if use_saved_images and transfer_analysis:
    # num of images that clean classified correctly, adversarial misclassified
    num_ctaf_images = clean_T_adver_F_idx[0].size
    # num of images that misclassified for the above images
    num_tfer_images = (correct[clean_T_adver_F_idx]==False).sum()
    if num_ctaf_images != 0:
      tfer_rate = 100*num_tfer_images/float(num_ctaf_images)
    else:
      tfer_rate = 0.
    print ('Transfer rate: %.1f (%d/%d)' % (tfer_rate, num_tfer_images,
                                            num_ctaf_images))

    # num of images that clean classified correctly for both source and target
    # networks, and adversarial misclassified
    num_ctaf_images2 = both_clean_T_adver_F_idx[0].size
    # num of images that misclassified for the above images
    num_tfer_images2 = (correct[both_clean_T_adver_F_idx]==False).sum()
    if num_ctaf_images2 != 0:
      tfer_rate2 = 100*num_tfer_images2/float(num_ctaf_images2)
    else:
      tfer_rate2 = 0.
    print ('2nd Transfer rate: %.1f (%d/%d)' % (tfer_rate2, num_tfer_images2,
                                                num_ctaf_images2))

  # Plot some examples of mis-classifications, if desired.
  if show_example_errors:
    print("Example errors:")
    plot_example_errors(test_images, cls_true, cls_pred, correct)

  # Plot the confusion matrix, if desired.
  if show_confusion_matrix:
    print("Confusion Matrix:")
    plot_confusion_matrix(cls_true, cls_pred)

def get_adver_grads(sess, placeholder_dict, tensor_dict,
                    data_from='validation',
                    batch_size=1,
                    custom_data=None,
                    custom_labels=None):
  """Get gradient w.r.t. the images.
     If custom_data is not None, we use custom_data instead of global dataset.

  Args:
    sess: tensorflow session
    placeholder_dict: dictionary of tensorflow placeholders
    tensor_dict: dictionary of tensorflow nodes
    data_from: string to indicate target dataset
               'training': training dataset
               'validation': validation dataset
               'test': test dataset
    batch_size: mini batch size for session run
    custom_data: if custom_data is not None, we use this as the input images
                 to get gradients instead of images from dataset.
    custom_labels: if custom_labels is not None, we use this as the input labels
                   to get gradients instead of labels from dataset.
  Returns:
    adver_grads: a numpy array of gradients w.r.t. the images.
  """

  x_adver = placeholder_dict['x_adver']
  y_adver = placeholder_dict['y_adver']
  option_adver = placeholder_dict['option_adver']

  grad_x_adver = tensor_dict['grad_x_adver']

  target_data = dataset_dict[data_from]

  if custom_data is not None:
    target_images = np.copy(custom_data)
    num_test = len(target_images)
    target_labels = custom_labels
  else:
    target_images, start_xy, flip_flag = copy_crop_images(target_data.images,
                                                          IMG_SIZE)
    num_test = len(target_images)
    target_labels = target_data.labels[0:num_test]

  adver_grads = np.empty_like(target_images)

  i = 0
  while i < num_test:
    if num_test-i < batch_size:
      j = num_test
    else:
      j = i + batch_size

    images = target_images[i:j, :]
    labels = target_labels[i:j, :]
    feed_dict_train = {x_adver: images, y_adver: labels,
                       option_adver: False}
    adver_grad = sess.run(grad_x_adver, feed_dict=feed_dict_train)
    adver_grads[i:j, :] = adver_grad
    i = j

  return adver_grads

def plot_save_corr_grads(test_e, per_image_corr_dict,
                    prefix, show_legend, show_flag,
                    show_annotation=True):
  """Plot and save the correlation between gradient w.r.t clean
     and gradient w.r.t adversarial images.
     (1) clean vs. step_ll
     (2) clean vs. step_FGSM
     (3) clean vs. random sign
     (4) step_ll vs. step_FGSM

  Args:
    test_e: list of epsilons
    per_image_corr_dict: a dictionary of mean and std of correlation.
                         Each key represents
                         'll_clean': clean vs. step_ll
                         'fgsm_clean': clean vs. step_FGSM
                         'rand_clean': clean vs. random sign
                         'll_fgsm': step_ll vs. step_FGSM
    prefix: string for prefix of file name to be saved.
    show_legend: True to show legend in graph.
    show_flag: True to show while running.
    show_annotation: True to show annotation in graph.
  """

  scale = 0.5
  ll_mean = per_image_corr_dict['ll_clean'][:,0]
  ll_std = scale*per_image_corr_dict['ll_clean'][:,1]
  fgsm_mean = per_image_corr_dict['fgsm_clean'][:,0]
  fgsm_std = scale*per_image_corr_dict['fgsm_clean'][:,1]
  rand_mean = per_image_corr_dict['rand_clean'][:,0]
  rand_std = scale*per_image_corr_dict['rand_clean'][:,1]
  ll_fgsm_mean = per_image_corr_dict['ll_fgsm'][:,0]
  ll_fgsm_std = scale*per_image_corr_dict['ll_fgsm'][:,1]

  fig, ax = plt.subplots()
  ax.axvline(0.2, color='k', linestyle='--', linewidth=2.5)
  ax.semilogx(test_e, ll_mean, label='(1) step_ll, clean',
      linewidth=LINE_WIDTH, color='b')
  ax.fill_between(test_e, ll_mean-ll_std, ll_mean+ll_std,
      facecolor='b', alpha=0.2)
  ax.semilogx(test_e, fgsm_mean, label='(2) step_FGSM, clean',
      linewidth=LINE_WIDTH, color='r')
  ax.fill_between(test_e, fgsm_mean-fgsm_std, fgsm_mean+fgsm_std,
      facecolor='r', alpha=0.2)
  ax.semilogx(test_e, rand_mean, label='(3) random, clean',
      linewidth=LINE_WIDTH, color='g')
  ax.fill_between(test_e, rand_mean-rand_std, rand_mean+rand_std,
      facecolor='g', alpha=0.2)
  ax.semilogx(test_e, ll_fgsm_mean, label='(4) step_ll, step_FGSM',
      linewidth=LINE_WIDTH, color='m')
  ax.fill_between(test_e, ll_fgsm_mean-ll_fgsm_std, ll_fgsm_mean+ll_fgsm_std,
      facecolor='m', alpha=0.2)
  plt.grid(True, which="both")

  if show_legend:
    legend = ax.legend(loc='lower left', shadow=True)
    for label in legend.get_texts():
      label.set_fontsize(LEGEND_FONT_SIZE)
    for label in legend.get_lines():
      label.set_linewidth(LINE_WIDTH)  # the legend line width

  if show_annotation:
    ll_annote = float(np.squeeze(
        per_image_ll_corr[np.where(np.array(test_e) ==0.2),0]))
    str_annote = '%.2f' % ll_annote
    ax.annotate(str_annote, xy=(0.2, ll_annote), xytext=(0.02, 0.4),
        size=FONT_SIZE,
        arrowprops=dict(facecolor='black', shrink=0.03, width=2))
    fgsm_annote = float(np.squeeze(
        per_image_fgsm_corr[np.where(np.array(test_e) ==0.2),0]))
    str_annote = '%.2f' % fgsm_annote
    ax.annotate(str_annote, xy=(0.2, fgsm_annote), xytext=(0.02, 0.2),
        size=FONT_SIZE,
        arrowprops=dict(facecolor='black', shrink=0.03, width=2))
    rand_annote = float(np.squeeze(
        per_image_rand_corr[np.where(np.array(test_e) ==0.2),0]))
    str_annote = '%.2f' % rand_annote
    ax.annotate(str_annote, xy=(0.2, rand_annote), xytext=(2, 0.8),
        size=FONT_SIZE,
        arrowprops=dict(facecolor='black', shrink=0.03, width=2))
    ll_fgsm_annote = float(np.squeeze(
        per_image_ll_fgsm_corr[np.where(np.array(test_e) ==0.2),0]))
    str_annote = '%.2f' % ll_fgsm_annote
    ax.annotate(str_annote, xy=(0.2, ll_fgsm_annote), xytext=(2, 0.6),
        size=FONT_SIZE,
        arrowprops=dict(facecolor='black', shrink=0.03, width=2))

  plt.ylim(-0.2, 1)
  plt.xlabel(r'$\epsilon$', fontsize=FONT_SIZE+8)
  plt.ylabel('correlation', fontsize=FONT_SIZE)
  for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(FONT_SIZE)
  for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(FONT_SIZE)
  figname = FLAGS.train_dir + '/' + prefix + '.pdf'
  plt.savefig(figname)
  if show_flag:
    plt.show(block=False)

def analyze_corr_grads(sess, placeholder_dict, tensor_dict,
                   data_from='validation',
                   test_idx=[0], sparsity=100, show_flag=False):
  """Plot and save the correlation between gradient w.r.t clean
     and gradient w.r.t adversarial images.
     (1) clean vs. step_ll
     (2) clean vs. step_FGSM
     (3) clean vs. random sign
     (4) step_ll vs. step_FGSM

  Args:
    sess: tensorflow session
    placeholder_dict: dictionary of tensorflow placeholders
    tensor_dict: dictionary of tensorflow nodes
    data_from: string to indicate target dataset
               'training': training dataset
               'validation': validation dataset
               'test': test dataset
    test_idx: list of index to be analized.
    sparsity: value in range [0, 100] to control sparsity of perturbation.
              If sparsity < 100, only sparsity (%) of pixels have non-zero
              adversarial noises.
    show_flag: True to show correlation plot.

  """

  batch_size = FLAGS.batch_size

  num_test = len(test_idx) # number of images to be tested
  # epsilons to be tested.
  test_e = [0.01]
  test_e += [e/100. for e in range(2, 100, 2)]
  test_e += [e/100. for e in range(100, 1020, 20)]

  # Define pair of images for correlation analysis.
  # The first column will have mean, the second column will have std.
  pairs = ['ll_clean', 'fgsm_clean', 'rand_clean', 'll_fgsm']
  first_args = ['step_ll', 'step_fgsm', 'step_rand', 'step_ll']
  second_args = ['clean', 'clean', 'clean', 'step_fgsm']
  per_image_corr_dict = {pair: np.zeros((len(test_e), 2)) for pair in pairs}

  target_data = dataset_dict[data_from]

  if num_test == 1:
    target_images, start_xy, flip_flag = copy_crop_images(
        np.expand_dims(target_data.images[test_idx], axis=0), IMG_SIZE)
    target_labels = target_data.labels[test_idx,:]
    cls_true = np.argmax(target_labels)
  else:
    target_images, start_xy, flip_flag = copy_crop_images(
        target_data.images[test_idx], IMG_SIZE)
    target_labels = target_data.labels[test_idx,:]
    cls_true = np.argmax(target_labels, axis=1)

  print ('number of test for gradients: %d' % num_test)
  print ('test index:')
  print (test_idx)
  print ('test label:')
  print (cls_true)

  unrolled_grads_dict = {}
  # get the gradients w.r.t. the clean images
  clean_grads = get_adver_grads(
      sess=sess,
      placeholder_dict=placeholder_dict,
      tensor_dict=tensor_dict,
      batch_size=batch_size,
      custom_data=target_images,
      custom_labels=target_labels)

  unrolled_grads_dict['clean'] = clean_grads.reshape((num_test, -1))

  # get the gradients w.r.t. adversarial images
  # We fix the rand_seed for random sign generation.
  methods = ['step_ll', 'step_fgsm', 'step_rand']
  rand_seeds = [None, None, 1234]
  adv_noises_dict = {}

  for method, rand_seed in zip(methods, rand_seeds):
    adv_images, _ = get_data(
        sess=sess,
        placeholder_dict=placeholder_dict,
        tensor_dict=tensor_dict,
        batch_size=batch_size,
        max_e=1,
        method=method,
        sparsity=sparsity,
        custom_data=target_images,
        custom_labels=target_labels,
        rand_seed=rand_seed)
    adv_noises_dict[method] = adv_images - target_images

  for i, max_e in enumerate(test_e):
    for method, adv_noises in adv_noises_dict.iteritems():
      adv_images = target_images + max_e*adv_noises
      grads = get_adver_grads(
          sess=sess,
          placeholder_dict=placeholder_dict,
          tensor_dict=tensor_dict,
          batch_size=batch_size,
          custom_data=adv_images,
          custom_labels=target_labels)

      unrolled_grads_dict[method] = grads.reshape((num_test, -1))

    for pair, first, second in zip(pairs, first_args, second_args):
      per_image_corrs = map(lambda x,y:np.corrcoef(x,y)[0,1],
                            unrolled_grads_dict[first],
                            unrolled_grads_dict[second])
      per_image_corr_dict[pair][i,0] = np.mean(per_image_corrs)
      per_image_corr_dict[pair][i,1] = np.std(per_image_corrs)

  plot_save_corr_grads(test_e, per_image_corr_dict,
                      'per_image_corr_'+str(sparsity), True, show_flag, False)


def plot_save_embeddings(test_e, adv_embeddings,
                         prefix, show_legend, show_flag):
  """Plot and save the sample embeddings for adversarial images with
     different epsilons.

  Args:
    test_e: list of epsilons
    adv_embeddings: a numpy array of mean/std of embeddings.
    prefix: string for prefix of file name to be saved.
    show_legend: True to show legend in graph.
    show_flag: True to show while running.
  """

  scale = 0.5
  true_mean = adv_embeddings[:,0]
  true_std = adv_embeddings[:,1]
  false_mean = adv_embeddings[:,2]
  false_std = adv_embeddings[:,3]
  fig, ax = plt.subplots()
  ax.plot(test_e, true_mean, label='True class', linewidth=LINE_WIDTH,
      color='b')
  ax.fill_between(test_e, true_mean-true_std, true_mean+true_std,
      facecolor='b', alpha=0.2)
  ax.plot(test_e, false_mean, label='False class', linewidth=LINE_WIDTH,
      color='r')
  ax.fill_between(test_e, false_mean-false_std, false_mean+false_std,
      facecolor='r', alpha=0.2)
  plt.grid()
  plt.ylim(-10, 25)
  if show_legend:
    legend = ax.legend(loc='upper right', shadow=True)
    for label in legend.get_texts():
      label.set_fontsize(LEGEND_FONT_SIZE)
    for label in legend.get_lines():
      label.set_linewidth(LINE_WIDTH)  # the legend line width
  plt.xlabel(r'$\epsilon$', fontsize=FONT_SIZE+8)
  plt.ylabel('argument to softmax', fontsize=FONT_SIZE)
  for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(FONT_SIZE)
  for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(FONT_SIZE)
  figname = FLAGS.train_dir + '/' + prefix + '.pdf'
  plt.savefig(figname)
  if show_flag:
    plt.show(block=False)

def visualize_embeddings(sess, placeholder_dict, tensor_dict,
                         data_from='validation',
                         test_idx=[0], sparsity=100, show_flag=False):
  """Plot and save the sample embeddings for adversarial images with
     different epsilons.

  Args:
    sess: tensorflow session
    placeholder_dict: dictionary of tensorflow placeholders
    tensor_dict: dictionary of tensorflow nodes
    data_from: string to indicate target dataset
               'training': training dataset
               'validation': validation dataset
               'test': test dataset
    test_idx: list of index to be analized.
    sparsity: value in range [0, 100] to control sparsity of perturbation.
              If sparsity < 100, only sparsity (%) of pixels have non-zero
              adversarial noises.
    show_flag: True to show correlation plot.

  """

  batch_size = FLAGS.batch_size
  num_test = len(test_idx) # number of images to be tested

  target_data = dataset_dict[data_from]

  x = placeholder_dict['x']
  phase = placeholder_dict['phase']
  embeddings = tensor_dict['embeddings']

  # epsilons to be tested.
  test_e = [-60, -55, -50, -45, -40, -35, -30,
            -28, -26, -24, -22, -20, -18, -16, -14, -12, -10,
            -9, -8, -7, -6, -5, -4, -3, -2, -1,
            -0.3, -0.1, -0.03, -0.01, -0.003, -0.001,
            0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1,
            2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20,
            22, 24, 26, 28, 30, 35, 40, 45, 50, 55, 59.9]

  if num_test == 1:
    target_images, start_xy, flip_flag = copy_crop_images(
        np.expand_dims(target_data.images[test_idx], axis=0), IMG_SIZE)
    target_labels = target_data.labels[test_idx,:]
    cls_true = np.argmax(target_labels)
  else:
    target_images, start_xy, flip_flag = copy_crop_images(
        target_data.images[test_idx], IMG_SIZE)
    target_labels = target_data.labels[test_idx,:]
    cls_true = np.argmax(target_labels, axis=1)

  print ('number of test for embeddings: %d' % num_test)
  print ('test index:')
  print (test_idx)
  print ('test label:')
  print (cls_true)

  methods = ['step_ll', 'step_fgsm', 'step_rand']
  rand_seeds = [None, None, 1234]
  adv_embeddings_dict = {method: np.zeros((len(test_e), 4))
                         for method in methods}

  for i, max_e in enumerate(test_e):
    for method, rand_seed in zip(methods, rand_seeds):
      adv_images, _ = get_data(
          sess=sess,
          placeholder_dict=placeholder_dict,
          tensor_dict=tensor_dict,
          batch_size=batch_size,
          max_e=max_e,
          method=method,
          sparsity=sparsity,
          custom_data=target_images,
          custom_labels=target_labels,
          rand_seed=rand_seed)

      adv_embeddings = sess.run(embeddings,
          feed_dict={x: adv_images, phase: False})
      true_embeddings = adv_embeddings[range(num_test), cls_true]
      false_embeddings = np.delete(adv_embeddings,
          [range(num_test), cls_true]).reshape((num_test, -1))

      adv_embeddings_dict[method][i,0] = true_embeddings.mean()
      adv_embeddings_dict[method][i,1] = np.std(true_embeddings)
      adv_embeddings_dict[method][i,2] = false_embeddings.mean()
      adv_embeddings_dict[method][i,3] = np.std(false_embeddings)

  for method in methods:
    plot_save_embeddings(test_e, adv_embeddings_dict[method],
                         method + '_' +str(sparsity), False, show_flag)

def trace_embeddings(sess, placeholder_dict, tensor_dict,
                     data_from='validation',
                     test_idx=[0], sparsity=100, show_flag=False):
  """Plot and save the scatter/trace plot of embeddings.

  Args:
    sess: tensorflow session
    placeholder_dict: dictionary of tensorflow placeholders
    tensor_dict: dictionary of tensorflow nodes
    data_from: string to indicate target dataset
               'training': training dataset
               'validation': validation dataset
               'test': test dataset
    test_idx: list of index to be analized.
    sparsity: value in range [0, 100] to control sparsity of perturbation.
              If sparsity < 100, only sparsity (%) of pixels have non-zero
              adversarial noises.
    show_flag: True to show correlation plot.

  """


  target_data = dataset_dict[data_from]

  x = placeholder_dict['x']
  phase = placeholder_dict['phase']
  embeddings = tensor_dict['embeddings']

  ##########################################################
  # First, all embedding
  ##########################################################
  batch_size = 100
  all_images, start_xy, flip_flag = copy_crop_images(target_data.images,
                                                     IMG_SIZE)
  all_labels = target_data.labels[:]

  num_test = len(all_images)

  # Allocate an array for the predicted classes which
  # will be calculated in batches and filled into this array.
  all_embeddings = np.zeros(shape=(num_test, 2), dtype=np.float)

  i = 0
  while i < num_test:
    if num_test-i < batch_size:
      j = num_test
    else:
      j = i + batch_size
    images = all_images[i:j, :]
    feed_dict = {x: images, phase: False}
    clean_embeddings = sess.run(embeddings, feed_dict=feed_dict)
    all_embeddings[i:j] = clean_embeddings[:,0:2]
    i = j

  ##########################################################
  # Next, sample adversarial embeddings
  ##########################################################
  num_test = len(test_idx) # number of images to be tested
  if FLAGS.data_set == 'mnist':
    test_e = range(0, 77, 4)
  elif 'cifar' in FLAGS.data_set:
    test_e = range(0, 17, 1)
  else:
    raise ValueError('Please choose between mnist, cifar10 and cifar100')

  methods = ['step_fgsm', 'iter_ll', 'iter_fgsm']
  sample_embeddings = {}

  if num_test == 1:
    target_images, start_xy, flip_flag = copy_crop_images(
        np.expand_dims(target_data.images[test_idx], axis=0), IMG_SIZE)
    target_labels = target_data.labels[test_idx,:]
    cls_true = np.argmax(target_labels)
  else:
    target_images, start_xy, flip_flag = copy_crop_images(
        target_data.images[test_idx], IMG_SIZE)
    target_labels = target_data.labels[test_idx,:]
    cls_true = np.argmax(target_labels, axis=1)

  print ('number of test for embeddings: %d' % num_test)
  print ('test index:')
  print (test_idx)
  print ('test label:')
  print (cls_true)
  rand_seed = 1234

  for method in methods:
    sample_embeddings[method] = []
    for max_e in test_e:
      # get the adversarial images
      adv_images, _ = get_data(
          sess=sess,
          placeholder_dict=placeholder_dict,
          tensor_dict=tensor_dict,
          batch_size=batch_size,
          max_e=max_e,
          method=method,
          sparsity=sparsity,
          custom_data=target_images,
          custom_labels=target_labels,
          rand_seed=rand_seed)

      adv_embeddings = sess.run(embeddings,
          feed_dict={x: adv_images, phase: False})
      sample_embeddings[method].append(adv_embeddings[:,0:2])

    markers = itertools.cycle(('<', '+', 'v', 'o', '^', 's', '*', 'x', 'D'))
    colors = itertools.cycle(('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                              '#bcbd22', '#17becf'))
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    array_embeddings = np.array(sample_embeddings[method])

    for i, label in enumerate(labels):
      color = colors.next()

      all_idx = np.where(np.argmax(all_labels, axis=1) == i)[0][:100]
      ax1.scatter(all_embeddings[all_idx,0], all_embeddings[all_idx,1],
               alpha=0.2, edgecolors='none',
               color=color)

      ax1.quiver(array_embeddings[:-1,i,0], array_embeddings[:-1,i,1],
                 array_embeddings[1:,i,0]-array_embeddings[:-1,i,0],
                 array_embeddings[1:,i,1]-array_embeddings[:-1,i,1],
          scale_units='xy', angles='xy', scale=1,
          color=color,
          linewidth=LINE_WIDTH, label=label)
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    if 'pivot' in FLAGS.train_dir:
      legend = ax1.legend(loc='center left', shadow=True,
          bbox_to_anchor=(1, 0.5))
      for label in legend.get_texts():
        label.set_fontsize(LEGEND_FONT_SIZE)
      for label in legend.get_lines():
        label.set_linewidth(LINE_WIDTH)

    for tick in ax1.xaxis.get_major_ticks():
      tick.label.set_fontsize(FONT_SIZE)
    for tick in ax1.yaxis.get_major_ticks():
      tick.label.set_fontsize(FONT_SIZE)
    figname = FLAGS.train_dir + '/' + data_from + '_' + method \
        + '_embeddings_trajectory.pdf'
    plt.savefig(figname)
    if show_flag:
      plt.show(block=False)


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
    name2var = dict(zip(map(lambda x: x.name.split(':')[0],
                            tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(sess, save_file)
