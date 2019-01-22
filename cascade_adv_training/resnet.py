import tensorflow as tf

from resnet_layers import *

n_dict = {3:20, 5:32, 7:44, 9:56, 18:110}
# ResNet architectures used for CIFAR-10
def resnet_e2(inpt, img_size, num_channels, num_classes, phase,
           n=3, embedding_at=2):

  layers = []

  with tf.variable_scope('conv1'):
    conv1 = gated_conv_layer(inpt, [3, 3, num_channels, 16], 1, phase)
    conv1 = tf.nn.relu(conv1)
    layers.append(conv1)

  if embedding_at == 1:
    nrof_embeddings = img_size*img_size*16
    embedding = tf.reshape(layers[-1], [-1, nrof_embeddings])

  for i in range (n):
    with tf.variable_scope('conv2_%d' % (i+1)):
      conv2 = gated_residual_block(layers[-1], 16, False, phase)
      layers.append(conv2)

    assert conv2.get_shape().as_list()[1:] == [img_size, img_size, 16]

  if embedding_at == 2:
    nrof_embeddings = img_size*img_size*16
    embedding = tf.reshape(layers[-1], [-1, nrof_embeddings])

  for i in range (n):
    down_sample = True if i == 0 else False
    with tf.variable_scope('conv3_%d' % (i+1)):
      conv3 = gated_residual_block(layers[-1], 32, down_sample, phase)
      layers.append(conv3)

    assert conv3.get_shape().as_list()[1:] == [img_size/2, img_size/2, 32]
  
  if embedding_at == 3:
    nrof_embeddings = img_size*img_size*8
    embedding = tf.reshape(layers[-1], [-1, nrof_embeddings])

  for i in range (n):
    down_sample = True if i == 0 else False
    with tf.variable_scope('conv4_%d' % (i+1)):
      conv4 = gated_residual_block(layers[-1], 64, down_sample, phase)
      layers.append(conv4)

    assert conv4.get_shape().as_list()[1:] == [img_size/4, img_size/4, 64]

  if embedding_at == 4:
    nrof_embeddings = img_size*img_size*4
    embedding = tf.reshape(layers[-1], [-1, nrof_embeddings])

  with tf.variable_scope('fc'):
    global_pool = tf.reduce_mean(layers[-1], [1, 2])
    assert global_pool.get_shape().as_list()[1:] == [64]

    fc = fc_layer(global_pool, [64, 2])
    layers.append(fc)

  with tf.variable_scope('fc2'):
    out = fc_layer(fc, [2, num_classes])
    layers.append(out)

  if embedding_at > 4:
    embedding = fc

  return embedding, layers[-1]

# ResNet architectures used for CIFAR-10
def resnet_e10x2(inpt, img_size, num_channels, num_classes, phase,
           n=3, embedding_at=2):

  layers = []

  with tf.variable_scope('conv1'):
    conv1 = gated_conv_layer(inpt, [3, 3, num_channels, 16], 1, phase)
    conv1 = tf.nn.relu(conv1)
    layers.append(conv1)

  if embedding_at == 1:
    nrof_embeddings = img_size*img_size*16
    embedding = tf.reshape(layers[-1], [-1, nrof_embeddings])

  for i in range (n):
    with tf.variable_scope('conv2_%d' % (i+1)):
      conv2 = gated_residual_block(layers[-1], 16, False, phase)
      layers.append(conv2)

    assert conv2.get_shape().as_list()[1:] == [img_size, img_size, 16]

  if embedding_at == 2:
    nrof_embeddings = img_size*img_size*16
    embedding = tf.reshape(layers[-1], [-1, nrof_embeddings])

  for i in range (n):
    down_sample = True if i == 0 else False
    with tf.variable_scope('conv3_%d' % (i+1)):
      conv3 = gated_residual_block(layers[-1], 32, down_sample, phase)
      layers.append(conv3)

    assert conv3.get_shape().as_list()[1:] == [img_size/2, img_size/2, 32]
  
  if embedding_at == 3:
    nrof_embeddings = img_size*img_size*8
    embedding = tf.reshape(layers[-1], [-1, nrof_embeddings])

  for i in range (n):
    down_sample = True if i == 0 else False
    with tf.variable_scope('conv4_%d' % (i+1)):
      conv4 = gated_residual_block(layers[-1], 64, down_sample, phase)
      layers.append(conv4)

    assert conv4.get_shape().as_list()[1:] == [img_size/4, img_size/4, 64]

  if embedding_at == 4:
    nrof_embeddings = img_size*img_size*4
    embedding = tf.reshape(layers[-1], [-1, nrof_embeddings])

  with tf.variable_scope('fc'):
    global_pool = tf.reduce_mean(layers[-1], [1, 2])
    assert global_pool.get_shape().as_list()[1:] == [64]

    fc = fc_layer(global_pool, [64, num_classes])
    layers.append(fc)

  with tf.variable_scope('fc2'):
    fc2 = fc_layer(fc, [num_classes, 2])
    layers.append(fc2)

  with tf.variable_scope('fc3'):
    out = fc_layer(fc2, [2, num_classes])
    layers.append(out)

  if embedding_at > 4:
    embedding = fc2

  return embedding, layers[-1]

# ResNet architectures used for CIFAR-10
def resnet(inpt, img_size, num_channels, num_classes, phase,
           n=3, embedding_at=2):

  layers = []

  with tf.variable_scope('conv1'):
    conv1 = gated_conv_layer(inpt, [3, 3, num_channels, 16], 1, phase)
    conv1 = tf.nn.relu(conv1)
    layers.append(conv1)

  if embedding_at == 1:
    nrof_embeddings = img_size*img_size*16
    embedding = tf.reshape(layers[-1], [-1, nrof_embeddings])

  for i in range (n):
    with tf.variable_scope('conv2_%d' % (i+1)):
      conv2 = gated_residual_block(layers[-1], 16, False, phase)
      layers.append(conv2)

    assert conv2.get_shape().as_list()[1:] == [img_size, img_size, 16]

  if embedding_at == 2:
    nrof_embeddings = img_size*img_size*16
    embedding = tf.reshape(layers[-1], [-1, nrof_embeddings])

  for i in range (n):
    down_sample = True if i == 0 else False
    with tf.variable_scope('conv3_%d' % (i+1)):
      conv3 = gated_residual_block(layers[-1], 32, down_sample, phase)
      layers.append(conv3)

    assert conv3.get_shape().as_list()[1:] == [img_size/2, img_size/2, 32]
  
  if embedding_at == 3:
    nrof_embeddings = img_size*img_size*8
    embedding = tf.reshape(layers[-1], [-1, nrof_embeddings])

  for i in range (n):
    down_sample = True if i == 0 else False
    with tf.variable_scope('conv4_%d' % (i+1)):
      conv4 = gated_residual_block(layers[-1], 64, down_sample, phase)
      layers.append(conv4)

    assert conv4.get_shape().as_list()[1:] == [img_size/4, img_size/4, 64]

  if embedding_at == 4:
    nrof_embeddings = img_size*img_size*4
    embedding = tf.reshape(layers[-1], [-1, nrof_embeddings])

  with tf.variable_scope('fc'):
    global_pool = tf.reduce_mean(layers[-1], [1, 2])
    assert global_pool.get_shape().as_list()[1:] == [64]
    
    out = fc_layer(global_pool, [64, num_classes])
    layers.append(out)
  if embedding_at > 4:
    embedding = out

  return embedding, layers[-1]

def noresnet(inpt, img_size, num_channels, num_classes, phase,
             n=3, embedding_at=2):

  layers = []

  with tf.variable_scope('conv1'):
    conv1 = gated_conv_layer(inpt, [3, 3, num_channels, 16], 1, phase)
    conv1 = tf.nn.relu(conv1)
    layers.append(conv1)

  if embedding_at == 1:
    nrof_embeddings = img_size*img_size*16
    embedding = tf.reshape(layers[-1], [-1, nrof_embeddings])

  for i in range (n):
    with tf.variable_scope('conv2_%d' % (i+1)):
      conv2 = gated_block(layers[-1], 16, False, phase)
      layers.append(conv2)

    assert conv2.get_shape().as_list()[1:] == [img_size, img_size, 16]

  if embedding_at == 2:
    nrof_embeddings = img_size*img_size*16
    embedding = tf.reshape(layers[-1], [-1, nrof_embeddings])

  for i in range (n):
    down_sample = True if i == 0 else False
    with tf.variable_scope('conv3_%d' % (i+1)):
      conv3 = gated_block(layers[-1], 32, down_sample, phase)
      layers.append(conv3)

    assert conv3.get_shape().as_list()[1:] == [img_size/2, img_size/2, 32]
  
  if embedding_at == 3:
    nrof_embeddings = img_size*img_size*8
    embedding = tf.reshape(layers[-1], [-1, nrof_embeddings])

  for i in range (n):
    down_sample = True if i == 0 else False
    with tf.variable_scope('conv4_%d' % (i+1)):
      conv4 = gated_block(layers[-1], 64, down_sample, phase)
      layers.append(conv4)

    assert conv4.get_shape().as_list()[1:] == [img_size/4, img_size/4, 64]

  if embedding_at == 4:
    nrof_embeddings = img_size*img_size*4
    embedding = tf.reshape(layers[-1], [-1, nrof_embeddings])

  with tf.variable_scope('fc'):
    global_pool = tf.reduce_mean(layers[-1], [1, 2])
    assert global_pool.get_shape().as_list()[1:] == [64]
    
    out = fc_layer(global_pool, [64, num_classes])
    layers.append(out)
  if embedding_at > 4:
    embedding = out

  return embedding, layers[-1]
