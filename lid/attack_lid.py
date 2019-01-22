import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import pickle
import keras.backend as K
from util import get_model
from extract_artifacts import get_lid
import collections
from detect_adv_samples import detect

import robustml


class Attack:
    def __init__(self, model, tol, num_steps, step_size, random_start):
        self.model = model
        self.tol = tol
        self.num_steps = num_steps
        self.step_size = step_size
        self.rand = random_start

        self.xs = tf.Variable(np.zeros((1000, 32, 32, 3), dtype=np.float32),
                                    name='modifier')
        self.orig_xs = tf.placeholder(tf.float32, [None, 32, 32, 3])

        self.ys = tf.placeholder(tf.int32, [None])

        self.epsilon = 8.0/255

        delta = tf.clip_by_value(self.xs, 0, 255) - self.orig_xs
        delta = tf.clip_by_value(delta, -self.epsilon, self.epsilon)

        self.do_clip_xs = tf.assign(self.xs, self.orig_xs+delta)

        self.logits = logits = model(self.xs)

        label_mask = tf.one_hot(self.ys, 10)
        correct_logit = tf.reduce_sum(label_mask * logits, axis=1)
        wrong_logit = tf.reduce_max((1-label_mask) * logits - 1e4*label_mask, axis=1)

        self.loss = (correct_logit - wrong_logit)

        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(step_size*1)

        grad,var = optimizer.compute_gradients(self.loss, [self.xs])[0]
        self.train = optimizer.apply_gradients([(tf.sign(grad),var)])

        end_vars = tf.global_variables()
        self.new_vars = [x for x in end_vars if x.name not in start_vars]

    def perturb(self, x, y, sess):
        sess.run(tf.variables_initializer(self.new_vars))
        sess.run(self.xs.initializer)
        sess.run(self.do_clip_xs,
                 {self.orig_xs: x})

        for i in range(self.num_steps):

            sess.run(self.train, feed_dict={self.ys: y})
            sess.run(self.do_clip_xs,
                     {self.orig_xs: x})

        return sess.run(self.xs)

provider = robustml.provider.CIFAR10("../cifar10_data/test_batch")
model = get_model("cifar", softmax=True)
model.load_weights("data/lid_model_cifar.h5")
model_logits = get_model("cifar", softmax=False)
model_logits.load_weights("data/lid_model_cifar.h5")
sess = K.get_session()
attack = Attack(model_logits,
                      1,
                      100,
                      1/255.0,
                      False)
xs = tf.placeholder(tf.float32, (1, 32, 32, 3))

x_input = tf.placeholder(tf.float32, [None, 32, 32, 3])
real_logits = model_logits(x_input)

succImages = 0
faillist = []
start = 0
end = 1000
totalImages = 0
outpath = 'perturb/lid_'
for i in range(start, end):
    outpath_pkl = outpath + str(i) + '.pkl'
    success = False
    inputs, targets= provider[i]
    logits = sess.run(real_logits, feed_dict={x_input: [inputs]})
    if np.argmax(logits) != targets:
        print('skip the wrong example ', i)
        continue
    totalImages += 1
    
    adversarial = attack.perturb([inputs], [targets], sess)
    
    if np.argmax(sess.run(real_logits, {x_input: adversarial[1:2]})) != targets:
        succImages += 1
        pickle.dump(adversarial[1:2],open(outpath_pkl,'wb'),-1)
        success = True
    if not success:
	    faillist.append(i)
print(faillist)
success_rate = succImages/float(totalImages)
print('succc rate', success_rate)
		
					  
