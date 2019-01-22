import numpy as np
import os
import tarfile
from six.moves import urllib
import sys


CIFAR10_DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CIFAR100_DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'

def maybe_download_and_extract(dest_directory, is_cifar10):
  """Download and extract the tarball from Alex's website."""
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  DATA_URL = CIFAR10_DATA_URL if is_cifar10 else CIFAR100_DATA_URL
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

  tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def one_hot(x, n):
  """
  convert index representation to one-hot representation
  """
  x = np.array(x)
  assert x.ndim == 1
  return np.eye(n)[x]


def _load_batch_cifar10(data_dir, filename, dtype='float32'):
  """
  load a batch in the CIFAR-10 format
  """
  #data_dir_cifar10 = os.path.join(data_dir, "cifar-10-batches-py")
  data_dir_cifar10 = data_dir
  path = os.path.join(data_dir_cifar10, filename)
  batch = np.load(path)
  data = batch['data']
  data = data.reshape([-1, 3, 32, 32])
  data = data.transpose([0, 2, 3, 1])
  # convert labels to one-hot representation
  labels = one_hot(batch['labels'], n=10)
  return data.astype(dtype), labels.astype(dtype)


def _grayscale(a):
  return np.expand_dims(a.reshape(-1, 32, 32, 3).mean(3), 3)
#  return a.reshape(a.shape[0], 32, 32, 3).mean(3).reshape(a.shape[0], -1)


def cifar10(data_dir, dtype='float32', grayscale=False):
  # train
  x_train = []
  t_train = []
  for k in range(5):
    x, t = _load_batch_cifar10(data_dir, "data_batch_%d" % (k + 1), dtype=dtype)
    x_train.append(x)
    t_train.append(t)

  x_train = np.concatenate(x_train, axis=0)
  t_train = np.concatenate(t_train, axis=0)

  # test
  x_test, t_test = _load_batch_cifar10(data_dir, "test_batch", dtype=dtype)

  if grayscale:
    x_train = _grayscale(x_train)
    x_test = _grayscale(x_test)

  return x_train, t_train, x_test, t_test


def _load_batch_cifar100(data_dir, filename, dtype='float32'):
  """
  load a batch in the CIFAR-100 format
  """
  data_dir_cifar100 = os.path.join(data_dir, "cifar-100-python")
  path = os.path.join(data_dir_cifar100, filename)
  batch = np.load(path)
  data = batch['data']
  data = data.reshape([-1, 3, 32, 32])
  data = data.transpose([0, 2, 3, 1])
  labels = one_hot(batch['fine_labels'], n=100)
  return data.astype(dtype), labels.astype(dtype)


def cifar100(data_dir, dtype='float32', grayscale=False):
  x_train, t_train = _load_batch_cifar100(data_dir, "train", dtype=dtype)
  x_test, t_test = _load_batch_cifar100(data_dir, "test", dtype=dtype)

  if grayscale:
    x_train = _grayscale(x_train)
    x_test = _grayscale(x_test)

  return x_train, t_train, x_test, t_test

