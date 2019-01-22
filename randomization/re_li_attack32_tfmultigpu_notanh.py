import robustml
from robustml_model import Randomization
import sys
import argparse
import tensorflow as tf
import numpy as np
from helpers import *
import time

import cv2

npop = 40 # 150 for titanxp     # population size
sigma = 0.1   # noise standard deviation
alpha = 0.008  # learning rate
# alpha = 0.001  # learning rate
boxmin = 0
boxmax = 1
boxplus = (boxmin + boxmax) / 2.
boxmul = (boxmax - boxmin) / 2.
folder = './liclipadvImages/'
import pickle
epsi = 0.031

samples = 8

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
    input_xs = tf.placeholder(tf.float32, [1, 299, 299, 3])
    input_img = tf.placeholder(tf.float32, [1, 299, 299, 3])
    modify_xs = tf.placeholder(tf.float32, [1, 32, 32, 3])
    real_logits_modify = model.batchlogits_modify_notanh(input_xs,modify_xs,input_img)
    real_logits_modify = tf.nn.softmax(real_logits_modify)
    input_xs_ori = tf.placeholder(tf.float32, [samples, 299, 299, 3])
    real_logits = model.batchlogits(input_xs_ori)
    real_logits = tf.nn.softmax(real_logits)

#     input_npopxs = tf.placeholder(tf.float32, [npop * samples, 299, 299, 3])
#     npop_logits = model.npoplogits(input_npopxs)

    input_npopxs = tf.placeholder(tf.float32, [1, 299, 299, 3])
    modify_tryxs = tf.placeholder(tf.float32, [npop, 32, 32, 3])
    npop_logits = model.multigpu_npoplogits_modify_notanh(input_npopxs,modify_tryxs,input_img)
    npop_logits = tf.nn.softmax(npop_logits)

    start = 150
    end = 600
    total = 0
    attacktime = 0
    #hardlist = [13, 23, 74, 77, 103, 104, 106, 115] # hard samples for npop = 40 samples = 4
    #hardlist = [152, 157, 159, 164, 165, 168, 170, 195, 198, 207, 210, 224, 234, 246, 255, 257, 263, 270, 279, 283, 303, 307, 318, 347, 356, 360, 367, 370,382, 387, 389, 402, 406, 413, 416, 443, 453, 459, 469, 470, 474, 536, 544, 546, 558, 566, 570, 575, 594]
    #hardlist = [152, 157, 165, 195, 224, 255, 257, 367, 370, 382, 387, 402, 469, 474, 544, 575]
    hardlist2 = [152, 157, 165, 195, 224, 255, 257, 367, 370, 382, 387, 402, 469, 474, 544, 575,637, 680, 688, 689, 692, 725, 730, 735, 745, 746, 751, 752, 765, 770, 774, 813, 816, 828, 829, 838, 841, 848]
    pklfolder = '/ssd/re_attack/pytorch-fcn/examples/voc/visualize/pickle_batch64/'
    for i in range(start, end):
        if i not in hardlist:
            continue
        success = False
        print('evaluating %d of [%d, %d)' % (i, start, end))
        inputs, targets= provider[i]
        modify = np.random.randn(1,32,32,3)*0.001 
#         modify = pickle.load(open(pklfolder+'ILSVRC2012_val_%08d'%(i+1)+'.pkl','rb'))
#         modify = modify.transpose(1,2,0).reshape(1,32,32,3)*0.01
#         print(np.mean(modify))
        print(modify.shape)
        inputs_expand = []
        for x in range(samples):
            inputs_expand.append(inputs)
        inputs_expand = np.array(inputs_expand)
        logits = sess.run(real_logits, feed_dict={input_xs_ori: inputs_expand})

        if np.argmax(logits) != targets:
            print('skip the wrong example ', i)
            print('max label {} , target label {}'.format(np.argmax(logits), targets),flush=True)
            continue

        totalImages += 1
        episode_start =time.time()

        newimg = np.arctanh((inputs-boxplus) / boxmul* 0.999999).reshape(1,299,299,3)
        for runstep in range(600):

            step_start = time.time()
            Nsample = np.random.randn(npop,32,32,3)

            modify_try = modify.repeat(npop,0) + sigma*Nsample
            temp = []
            resize_start =time.time()
#             for x in modify_try:
#                 temp.append(cv2.resize(x, dsize=(299,299), interpolation=cv2.INTER_LINEAR))
#             modify_try = np.array(temp)
#             print('resize time ', time.time()-resize_start,flush=True)


            if runstep % 10 == 0:
#                 temp = []
#                 for x in modify:
#                     temp.append(cv2.resize(x, dsize=(299,299), interpolation=cv2.INTER_LINEAR))
#                 modify_test = np.array(temp)


                outputsreal = sess.run(real_logits_modify, feed_dict={input_xs: newimg,modify_xs:modify,input_img:[inputs]})
                outputsreal = np.log(outputsreal)
                print('logits ',np.sort(outputsreal)[-1:-6:-1])
                print('target label ', np.argsort(outputsreal)[-1:-6:-1])
                print('negative_logits ', np.sort(outputsreal)[0:3:1])
                sys.stdout.flush()

#                 print(np.abs(realclipdist).max())

#                 if (np.argmax(outputsreal) != targets) and (np.abs(realclipdist).max() <= epsi):
                if (np.argmax(outputsreal) != targets):

                    attacktime += time.time()-episode_start
                    print('episode time : ', time.time()-episode_start,flush=True)
                    print('atack time : ', attacktime,flush=True)
                    succImages += 1
                    success = True
                    print('clipimage succImages: '+str(succImages)+'  totalImages: '+str(totalImages))
#                     print('lirealsucc: '+str(realclipdist.max()))
                    sys.stdout.flush()
                    break

            target_onehot =  np.zeros((1,1000))


            target_onehot[0][targets]=1.

            outputs = sess.run(npop_logits, feed_dict={input_npopxs:newimg,modify_tryxs:modify_try,input_img:[inputs]})

            outputs = np.log(outputs)
            target_onehot = target_onehot.repeat(npop,0)



            real = (target_onehot * outputs).sum(1)
            other = ((1. - target_onehot) * outputs - target_onehot * 10000.).max(1)[0]

            loss1 = np.clip(real - other, 0.,1000)

            Reward = 0.5 * loss1
#             Reward = l2dist

            Reward = -Reward

            A = (Reward - np.mean(Reward)) / (np.std(Reward)+1e-7)


            modify = modify + (alpha/(npop*sigma)) * ((np.dot(Nsample.reshape(npop,-1).T, A)).reshape(32,32,3))
#             print('one step time : ', time.time()-step_start)
        if not success:
            print('fail, ',faillist)
            faillist.append(i)
#         print('episode time : ', time.time()-episode_start,flush=True)
    print(faillist)
    success_rate = succImages/float(totalImages)




    print('attack success rate: %.2f%% (over %d data points)' % (success_rate*100, args.end-args.start))

if __name__ == '__main__':
    main()
