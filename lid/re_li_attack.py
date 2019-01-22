import robustml
import time
import pickle
from util import get_model
import sys
import argparse
import tensorflow as tf
import numpy as np
from helpers import *
import keras.backend as K



npop = 300     # population size
sigma = 0.1    # noise standard deviation
alpha = 0.008  # learning rate
# alpha = 0.001  # learning rate
boxmin = 0
boxmax = 1
boxplus = (boxmin + boxmax) / 2.
boxmul = (boxmax - boxmin) / 2.

epsi = 0.031
epsilon = 1e-30
def softmax(x):
        return np.divide(np.exp(x),np.sum(np.exp(x),-1,keepdims=True))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--cifar-path', type=str, default='../cifar10_data/test_batch',
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
    K.set_session(sess)
    #sess = tf.Session()

    # initialize a model
    model = get_model('cifar', softmax=False)
    ### change lid model path                          llj
    model.load_weights("../all_models/lid/lid_model_cifar.h5")


    # initialize a data provider for CIFAR-10 images
    provider = robustml.provider.CIFAR10(args.cifar_path)
    input_xs = tf.placeholder(tf.float32, [None, 32, 32, 3])

    real_logits0 = model(input_xs)
    real_logits = tf.nn.softmax(real_logits0)

    start = 0
    end = 100
    total = 0
    successlist = []
    printlist = []

    start_time = time.time()

    for i in range(start, end):
        success = False
        print('evaluating %d of [%d, %d)' % (i, start, end), file=sys.stderr)
        inputs, targets= provider[i]
    
        modify = np.random.randn(1,3,32,32) * 0.001
        ##### thermometer encoding

        logits = sess.run(real_logits, feed_dict={input_xs: [inputs]})
        print(logits)
        if np.argmax(logits) != targets:
            print('skip the wrong example ', i)
            continue
        totalImages += 1

        for runstep in range(400):
            Nsample = np.random.randn(npop, 3,32,32)

            modify_try = modify.repeat(npop,0) + sigma*Nsample

            newimg = torch_arctanh((inputs-boxplus) / boxmul).transpose(2,0,1)

            inputimg = np.tanh(newimg+modify_try) * boxmul + boxplus
            if runstep % 10 == 0:
                realinputimg = np.tanh(newimg+modify) * boxmul + boxplus
                realdist = realinputimg - (np.tanh(newimg) * boxmul + boxplus)
                realclipdist = np.clip(realdist, -epsi, epsi)
                realclipinput = realclipdist + (np.tanh(newimg) * boxmul + boxplus)
                l2real =  np.sum((realclipinput - (np.tanh(newimg) * boxmul + boxplus))**2)**0.5
                #l2real =  np.abs(realclipinput - inputs.numpy())
                print(inputs.shape)
                outputsreal = sess.run(real_logits, feed_dict={input_xs: realclipinput.transpose(0,2,3,1)})
                print(outputsreal)

                print(np.abs(realclipdist).max())
                print('l2real: '+str(l2real.max()))
                print(outputsreal)
                if (np.argmax(outputsreal) != targets) and (np.abs(realclipdist).max() <= epsi):
                    succImages += 1
                    success = True
                    print('clipimage succImages: '+str(succImages)+'  totalImages: '+str(totalImages))
                    print('lirealsucc: '+str(realclipdist.max()))
                    successlist.append(i)
                    printlist.append(runstep)
                    break
            dist = inputimg - (np.tanh(newimg) * boxmul + boxplus)
            clipdist = np.clip(dist, -epsi, epsi)
            clipinput = (clipdist + (np.tanh(newimg) * boxmul + boxplus)).reshape(npop,3,32,32)
            target_onehot =  np.zeros((1,10))


            target_onehot[0][targets]=1.

            outputs = sess.run(real_logits, feed_dict={input_xs: clipinput.transpose(0,2,3,1)})

            target_onehot = target_onehot.repeat(npop,0)



            real = np.log((target_onehot * outputs).sum(1)+1e-30)
            other = np.log(((1. - target_onehot) * outputs - target_onehot * 10000.).max(1)[0] + 1e-30)

            loss1 = np.clip(real - other, 0.,1000)

            Reward = 0.5 * loss1
#             Reward = l2dist

            Reward = -Reward

            A = (Reward - np.mean(Reward)) / (np.std(Reward)+1e-7)


            modify = modify + (alpha/(npop*sigma)) * ((np.dot(Nsample.reshape(npop,-1).T, A)).reshape(3,32,32))
        if not success:
            faillist.append(i)
            print('failed:',faillist)
        else:
            print('successed:',successlist)
    print(faillist)
    success_rate = succImages/float(totalImages)
    np.savez('runstep',printlist)
    end_time = time.time()
    print('all time :', end_time - start_time)
    print('succc rate', success_rate)


#     print('attack success rate: %.2f%% (over %d data points)' % (success_rate*100, args.end-args.start))

if __name__ == '__main__':
    main()
