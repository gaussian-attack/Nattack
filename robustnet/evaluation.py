import robustml
import cv2
from utils import *
import sys
import argparse
import numpy as np
from helpers import *
#import keras.backend as K
import torch
from torch.autograd import Variable
import torch.optim as optim
import torchvision.transforms as tfs
import torchvision.datasets as dst
from torch.utils.data import DataLoader
import torch.nn as nn
from models import *
import torch.autograd as autograd
import pickle
import time
import os


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


def softmax(x):
    return np.divide(np.exp(x),np.sum(np.exp(x),-1,keepdims=True))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1 and m.affine:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--cifar-path', type=str, default='../cifar10_data/test_batch',
            help='path to the test_batch file from http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
    parser.add_argument('--perturb', type=str, default='perturb')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--noise', type=float, default=0.3)
    parser.add_argument('--modelIn', type=str, default='vgg16/noise_0.3.pth')
    args = parser.parse_args()

    test_loss = 0
    correct = 0
    total = 0
    totalImages = 0
    succImages = 0
    faillist = []

    model = VGG("VGG16", args.noise)
    model = nn.DataParallel(model, device_ids=range(1))
    loss_f = nn.CrossEntropyLoss()
    model.apply(weights_init)
    if args.modelIn is not None:
        model.load_state_dict(torch.load(args.modelIn))
        print('load successfully', flush=True)
    model.cuda()
    model.eval()

    # initialize an attack (it's a white box attack, and it's allowed to look
    # at the internals of the model in any way it wants)
    # attack = BPDA(sess, model, epsilon=model.threat_model.epsilon, debug=args.debug)
    # attack = Attack(sess, model.model, epsilon=model.threat_model.epsilon)

    # initialize a data provider for CIFAR-10 images
    provider = robustml.provider.CIFAR10(args.cifar_path)
    means = autograd.Variable(
        torch.from_numpy(np.array([0.4914, 0.4822, 0.4465]).reshape([1, 3, 1, 1]).astype('float32')).cuda(),
        volatile=True)
    # means = torch.tensor(np.array([0.4914,0.4822,0.4465]).reshape(1,3,1,1)).float()
    # stds = torch.tensor(np.array([0.2023, 0.1994, 0.2010]).reshape(1,3,1,1)).float()
    stds = autograd.Variable(
        torch.from_numpy(np.array([0.2023, 0.1994, 0.2010]).reshape([1, 3, 1, 1]).astype('float32')).cuda(),
        volatile=True)

    start = 0
    end = 10000
    total = 0
    successlist = []
    printlist = []

    start_time = time.time()
    perturbs = os.listdir('./')
    all_dir = []
    for x in perturbs:
        if 'perturb' in x:
            all_dir.append(x)


    for y in all_dir:
      perturb_files = os.listdir(y)
      numbers = []
      totalImages = 0
      succImages = 0

      numbers = []
      for x in perturb_files:
        number = x.split('_')[1]
        name = x.split('_')[0]
        number1 = int(number.split('.pkl')[0])
        numbers.append(number1)

      for i in numbers:
        success = False
        print('evaluating %d of [%d, %d)' % (i, start, end), file=sys.stderr)
        inputs, targets= provider[i]

        in_pkl = y + '/' + name + '_' + str(i)+'.pkl'
        ##### thermometer encoding

        input_var = autograd.Variable(torch.from_numpy(inputs.transpose(2, 0, 1)).cuda(), volatile=True)

        logits = model((input_var - means) / stds).data.cpu().numpy()
        probs = softmax(logits)

        if np.argmax(probs[0]) != targets:
            print('skip the wrong example ', i)
            continue
        totalImages += 1
        try:
            modify = pickle.load(open(in_pkl, 'rb'))
        except:
            modify = pickle.load(open(in_pkl, 'rb'),encoding='bytes')
        if 'cascade' in y:
            modify = cv2.resize(modify[0].transpose(1,2,0), dsize=(32,32), interpolation=cv2.INTER_LINEAR)
            modify = modify.reshape((1,3,32,32))
        newimg = torch_arctanh((inputs - boxplus) / boxmul).transpose(2, 0, 1)
        realinputimg = np.tanh(newimg + modify) * boxmul + boxplus
        realdist = realinputimg - (np.tanh(newimg) * boxmul + boxplus)
        realclipdist = np.clip(realdist, -epsi, epsi)
        realclipinput = realclipdist + (np.tanh(newimg) * boxmul + boxplus)
        input_var = autograd.Variable(torch.from_numpy(realclipinput.astype('float32')).cuda(), volatile=True)

        outputsreal = model((input_var - means) / stds).data.cpu().numpy()[0]
        outputsreal = softmax(outputsreal)
        if (np.argmax(outputsreal) != targets) and (np.abs(realclipdist).max() <= epsi):
            succImages += 1

        success_rate = succImages / float(totalImages)
      print('name:',y)
      print('succ rate', success_rate)
      print('succ {} , total {}'.format(succImages, totalImages))


if __name__ == '__main__':
    main()
