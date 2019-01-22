import robustml
from robustml_model import Randomization
import sys
import argparse
import tensorflow as tf
import numpy as np
from helpers import *
import time

import cv2

npop = 360 # 150 for titanxp     # population size
sigma = 0.1    # noise standard deviation
alpha = 0.008  # learning rate
# alpha = 0.001  # learning rate
boxmin = 0
boxmax = 1
boxplus = (boxmin + boxmax) / 2.
boxmul = (boxmax - boxmin) / 2.
folder = './liclipadvImages/'
epsi = 0.031

samples =4

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--imagenet-path', type=str, default='../../obfuscated_zoo/imagenet_val',
            help='path to the test_batch file from http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    test_loss = 0
    correct = 0
    total = 0
    totalImages = 0
    succImages = 0
    faillist = []



    # set up TensorFlow session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)

    # initialize a model
    model = Randomization(sess)

    print(model.threat_model.targeted)
    # initialize an attack (it's a white box attack, and it's allowed to look
    # at the internals of the model in any way it wants)
    # attack = BPDA(sess, model, epsilon=model.threat_model.epsilon, debug=args.debug)
    # attack = Attack(sess, model.model, epsilon=model.threat_model.epsilon)

    # initialize a data provider for CIFAR-10 images
    provider = robustml.provider.ImageNet(args.imagenet_path, model.dataset.shape)
    input_xs = tf.placeholder(tf.float32, [samples, 299, 299, 3])
    real_logits = model.batchlogits(input_xs)

    input_npopxs = tf.placeholder(tf.float32, [npop * samples, 299, 299, 3])
    npop_logits = model.multigpu_npoplogits(input_npopxs)

    start = 0
    end = 140
    total = 0
    attacktime = 0
    for i in range(start, end):
        success = False
        print('evaluating %d of [%d, %d)' % (i, start, end))
        inputs, targets= provider[i]
        modify = np.random.randn(1,3,32,32) * 0.001
        inputs_expand = []
        for x in range(samples):
            inputs_expand.append(inputs)
        inputs_expand = np.array(inputs_expand)
        logits = sess.run(real_logits, feed_dict={input_xs: inputs_expand})
        # print(logits)
        if np.argmax(logits) != targets:
            print('skip the wrong example ', i)
            print('max label {} , target label {}'.format(np.argmax(logits), targets),flush=True)
            continue
        totalImages += 1
        episode_start =time.time()


        for runstep in range(500):

            step_start = time.time()
            Nsample = np.random.randn(npop, 3,32,32)

            modify_try = modify.repeat(npop,0) + sigma*Nsample
            temp = []
            resize_start =time.time()
            for x in modify_try:
                temp.append(cv2.resize(x.transpose(1,2,0), dsize=(299,299), interpolation=cv2.INTER_LINEAR).transpose(2,0,1))
            modify_try = np.array(temp)
#             print('resize time ', time.time()-resize_start,flush=True)
            #modify_try = cv2.resize(modify_try.transpose(0,2,3,1), dsize=(299, 299), interpolation=cv2.INTER_CUBIC).transpose(0,3,1,2)
            #print(modify_try.shape, flush=True)

            newimg = torch_arctanh((inputs-boxplus) / boxmul).transpose(2,0,1)

            inputimg = np.tanh(newimg+modify_try) * boxmul + boxplus
            if runstep % 10 == 0:
                temp = []
                for x in modify:
                    temp.append(cv2.resize(x.transpose(1,2,0), dsize=(299,299), interpolation=cv2.INTER_LINEAR).transpose(2,0,1))
                modify_test = np.array(temp)

                #modify_test = cv2.resize(modify.transpose(0,2,3,1), dsize=(299, 299), interpolation=cv2.INTER_CUBIC).transpose(0,3,1,2)
                realinputimg = np.tanh(newimg+modify_test) * boxmul + boxplus
                realdist = realinputimg - (np.tanh(newimg) * boxmul + boxplus)
                realclipdist = np.clip(realdist, -epsi, epsi)
                realclipinput = realclipdist + (np.tanh(newimg) * boxmul + boxplus)
                l2real =  np.sum((realclipinput - (np.tanh(newimg) * boxmul + boxplus))**2)**0.5
                #l2real =  np.abs(realclipinput - inputs.numpy())


                realclipinput = realclipinput.transpose(0, 2, 3, 1)
                realclipinput = np.squeeze(realclipinput)
                realclipinput_expand = []
                for x in range(samples):
                    realclipinput_expand.append(realclipinput)
                realclipinput_expand = np.array(realclipinput_expand)

                outputsreal = sess.run(real_logits, feed_dict={input_xs: realclipinput_expand})

                print('logits ',np.sort(outputsreal)[-1:-6:-1])
                print('target label ', np.argsort(outputsreal)[-1:-6:-1])
                print('negative_logits ', np.sort(outputsreal)[0:3:1])
                sys.stdout.flush()
                # print(outputsreal)

                #print(np.abs(realclipdist).max())
                #print('l2real: '+str(l2real.max()))
                # print(outputsreal)
                if (np.argmax(outputsreal) != targets) and (np.abs(realclipdist).max() <= epsi):
                    attacktime += time.time()-episode_start
                    print('episode time : ', time.time()-episode_start,flush=True)
                    print('atack time : ', attacktime,flush=True)
                    succImages += 1
                    success = True
                    print('clipimage succImages: '+str(succImages)+'  totalImages: '+str(totalImages))
                    print('lirealsucc: '+str(realclipdist.max()))
                    sys.stdout.flush()
#                     imsave(folder+classes[targets[0]]+'_'+str("%06d" % batch_idx)+'.jpg',inputs.transpose(1,2,0))
                    break
            dist = inputimg - (np.tanh(newimg) * boxmul + boxplus)
            clipdist = np.clip(dist, -epsi, epsi)
            clipinput = (clipdist + (np.tanh(newimg) * boxmul + boxplus)).reshape(npop,3,299,299)
            target_onehot =  np.zeros((1,1000))


            target_onehot[0][targets]=1.


            input_start = time.time()
            clipinput = clipinput.transpose(0, 2, 3, 1)
            clipinput = np.squeeze(clipinput)
            clipinput_expand = []
            for x in range(samples):
                clipinput_expand.append(clipinput)
            clipinput_expand = np.array(clipinput_expand)
            clipinput_expand = clipinput_expand.reshape((samples * npop, 299, 299, 3))

            outputs = sess.run(npop_logits, feed_dict={input_npopxs: clipinput_expand})
#             print('input_time : ', time.time()-input_start,flush=True)

            target_onehot = target_onehot.repeat(npop,0)



            real = (target_onehot * outputs).sum(1)
            other = ((1. - target_onehot) * outputs - target_onehot * 10000.).max(1)[0]

            loss1 = np.clip(real - other, 0.,1000)

            Reward = 0.5 * loss1
#             Reward = l2dist

            Reward = -Reward

            A = (Reward - np.mean(Reward)) / (np.std(Reward)+1e-7)


            modify = modify + (alpha/(npop*sigma)) * ((np.dot(Nsample.reshape(npop,-1).T, A)).reshape(3,32,32))
#             print('one step time : ', time.time()-step_start)
        if not success:
            faillist.append(i)
#         print('episode time : ', time.time()-episode_start,flush=True)
    print(faillist)
    success_rate = succImages/float(totalImages)




    print('attack success rate: %.2f%% (over %d data points)' % (success_rate*100, args.end-args.start))

if __name__ == '__main__':
    main()
