import robustml
from defense import defend
from inceptionv3 import model as inceptionv3_model
import tensorflow as tf
from defense_batch import defend_batch

npop = 40 # 150 for titan xp
samples = 8

gpu_num = 4

class Randomization(robustml.model.Model):
    def __init__(self, sess):
        self._sess = sess
        self._input = tf.placeholder(tf.float32, (299, 299, 3))
        input_expanded = tf.expand_dims(self._input, axis=0)
        randomized = defend(input_expanded)
        self._logits, self._predictions = inceptionv3_model(sess, randomized)
        self._dataset = robustml.dataset.ImageNet((299, 299, 3))
        self._threat_model = robustml.threat_model.Linf(epsilon=8.0/255.0)

        boxmin = 0
        boxmax = 1
        self.boxplus = (boxmin + boxmax) / 2.
        self.boxmul = (boxmax - boxmin) / 2.
        self.epsi = 0.031

    @property
    def dataset(self):
        return self._dataset

    @property
    def threat_model(self):
        return self._threat_model

    def classify(self, x):
        return self._sess.run(self._predictions, {self._input: x})[0]

    # expose internals for white box attacks

    def outlogits(self, x):
        randomized = defend(x)
        logits, _ = inceptionv3_model(self._sess, randomized)
        return logits

    def npoplogits(self, x):
        a = []
        for i in range(x.shape[0]):
            temp_randomized = defend(x[i:i+1])
            temp_randomized = tf.squeeze(temp_randomized)
            a.append(temp_randomized)
        randomized = tf.convert_to_tensor(a)
        logits, _ = inceptionv3_model(self._sess, randomized)

        logits = tf.reshape(logits, [samples, npop, -1])

        logits = tf.reduce_mean(logits, axis=0)
        return logits

    def batchlogits(self, x):
        a = []
        for i in range(x.shape[0]):
            temp_randomized = defend(x[i:i+1])
            temp_randomized = tf.squeeze(temp_randomized)
            a.append(temp_randomized)
        randomized = tf.convert_to_tensor(a)
        logits, _ = inceptionv3_model(self._sess, randomized)
        logits = tf.reduce_mean(logits, axis=0)
        return logits

    def multigpu_npoplogits(self, inputx):
        x = inputx

        x_a = tf.split(x, gpu_num)
        final_logits = []
        for gpu_id in range(gpu_num):
            with tf.device('/gpu:%d' % gpu_id):
                a = []
                for i in range(x_a[gpu_id].shape[0]):
                    temp_randomized = defend(x_a[gpu_id][i:i + 1])
                    temp_randomized = tf.squeeze(temp_randomized)
                    a.append(temp_randomized)
                tensor = tf.convert_to_tensor(a)

                logits, _ = inceptionv3_model(self._sess, tensor)
                logits = tf.reshape(logits, [int(samples / gpu_num), npop, -1])
                final_logits.append(logits)
        final_logits = tf.convert_to_tensor(final_logits)
        final_logits = tf.reshape(final_logits, [samples, npop, -1])
        final_logits = tf.reduce_mean(final_logits, axis=0)
        return final_logits

    def multigpu_npoplogits_modify(self, inputx, modify_try,input_img):
        with tf.device('/gpu:0'): #### run on gpu 0    lllj ?
          resized_images = tf.image.resize_images(modify_try, [299, 299],method=0)
          ori_x = tf.tanh(inputx + resized_images) * self.boxmul + self.boxplus
        realdist = ori_x - input_img
        realclipdist = tf.clip_by_value(realdist, -self.epsi, self.epsi)
        ensemble_prex = realclipdist + input_img
        ensemble_x = tf.concat([ensemble_prex for _ in range(samples)], axis=0)
        x = tf.reshape(ensemble_x, [samples * npop, 299, 299, 3])

        x_a = tf.split(x, gpu_num)
        final_logits = []
        for gpu_id in range(gpu_num):
            with tf.device('/gpu:%d' % gpu_id):
                a = []
                for i in range(x_a[gpu_id].shape[0]):
                    temp_randomized = defend(x_a[gpu_id][i:i + 1])
                    temp_randomized = tf.squeeze(temp_randomized)
                    a.append(temp_randomized)
                tensor = tf.convert_to_tensor(a)

                logits, _ = inceptionv3_model(self._sess, tensor)
                logits = tf.reshape(logits, [int(samples / gpu_num), npop, -1])
                final_logits.append(logits)
        final_logits = tf.convert_to_tensor(final_logits)
        final_logits = tf.reshape(final_logits, [samples, npop, -1])
        final_logits = tf.reduce_mean(final_logits, axis=0)
        return final_logits
    def multigpu_npoplogits_modify_notanh(self, inputx, modify_try,input_img):
        with tf.device('/gpu:0'): #### run on gpu 0    lllj ?
          resized_images = tf.image.resize_images(modify_try, [299, 299],method=0)
          #ori_x = tf.tanh(inputx + resized_images) * self.boxmul + self.boxplus
        realdist = resized_images
        realclipdist = tf.clip_by_value(realdist, -self.epsi, self.epsi)
        ensemble_prex = realclipdist + input_img

        ensemble_prex = tf.clip_by_value(ensemble_prex, 0, 1)
        ensemble_x = tf.concat([ensemble_prex for _ in range(samples)], axis=0)
        
        x = tf.reshape(ensemble_x, [samples * npop, 299, 299, 3])

        x_a = tf.split(x, gpu_num)
        final_logits = []
        for gpu_id in range(gpu_num):
            with tf.device('/gpu:%d' % gpu_id):
                a = []
                for i in range(x_a[gpu_id].shape[0]):
                    temp_randomized = defend(x_a[gpu_id][i:i + 1])
                    temp_randomized = tf.squeeze(temp_randomized)
                    a.append(temp_randomized)
                tensor = tf.convert_to_tensor(a)

                logits, _ = inceptionv3_model(self._sess, tensor)
                logits = tf.reshape(logits, [int(samples / gpu_num), npop, -1])
                final_logits.append(logits)
        final_logits = tf.convert_to_tensor(final_logits)
        final_logits = tf.reshape(final_logits, [samples, npop, -1])
        final_logits = tf.reduce_mean(final_logits, axis=0)
        return final_logits



    def outlogits_modify(self, x, modify):
        randomized = defend(x)
        logits, _ = inceptionv3_model(self._sess, randomized)
        return logits

    def npoplogits_modify(self, inputx, modify_try):

        resized_images=tf.image.resize_images(modify_try, [299, 299], method=0)
        ori_x = tf.tanh(inputx + resized_images) * self.boxmul + self.boxplus
        realdist = ori_x - (tf.tanh(inputx) * self.boxmul + self.boxplus)
        realclipdist = tf.clip_by_value(realdist, -self.epsi, self.epsi)
        ensemble_prex = realclipdist + (tf.tanh(inputx) * self.boxmul + self.boxplus)
        ensemble_x = tf.concat([ensemble_prex for _ in range(samples)], axis=0)
        x = tf.reshape(ensemble_x, [samples * npop, 299, 299, 3])

        a = []
        for i in range(x.shape[0]):
            temp_randomized = defend(x[i:i+1])
            temp_randomized = tf.squeeze(temp_randomized)
            a.append(temp_randomized)
        randomized = tf.convert_to_tensor(a)
        logits, _ = inceptionv3_model(self._sess, randomized)

        logits = tf.reshape(logits, [samples, npop, -1])

        logits = tf.reduce_mean(logits, axis=0)
        return logits
    def batchlogits_modify_notanh(self, inputx, modify,input_img):
        resized_images=tf.image.resize_images(modify, [299, 299], method=0)
#         ori_x = tf.tanh(inputx+resized_images) * self.boxmul + self.boxplus
        realdist = resized_images
        realclipdist = tf.clip_by_value(realdist, -self.epsi, self.epsi)
        ensemble_prex = realclipdist + input_img
        ensemble_prex = tf.clip_by_value(ensemble_prex, 0, 1)
        x = tf.concat([ensemble_prex for _ in range(samples)], axis=0)
        a = []
        for i in range(x.shape[0]):
            temp_randomized = defend(x[i:i+1])
            temp_randomized = tf.squeeze(temp_randomized)
            a.append(temp_randomized)
        randomized = tf.convert_to_tensor(a)
        logits, _ = inceptionv3_model(self._sess, randomized)
        logits = tf.reduce_mean(logits, axis=0)
        return logits

    def batchlogits_modify(self, inputx, modify,input_img):
        resized_images=tf.image.resize_images(modify, [299, 299], method=0)
        ori_x = tf.tanh(inputx+resized_images) * self.boxmul + self.boxplus
        realdist = ori_x - input_img
        realclipdist = tf.clip_by_value(realdist, -self.epsi, self.epsi)
        ensemble_prex = realclipdist + input_img
        x = tf.concat([ensemble_prex for _ in range(samples)], axis=0)
        a = []
        for i in range(x.shape[0]):
            temp_randomized = defend(x[i:i+1])
            temp_randomized = tf.squeeze(temp_randomized)
            a.append(temp_randomized)
        randomized = tf.convert_to_tensor(a)
        logits, _ = inceptionv3_model(self._sess, randomized)
        logits = tf.reduce_mean(logits, axis=0)
        return logits

    @property
    def input(self):
        return self._input

    @property
    def logits(self):
        return self._logits

    @property
    def predictions(self):
        return self._predictions
