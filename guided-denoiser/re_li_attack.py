#from __future__ import print_function

from provider import ImageNet
import os
import argparse
import math
import numpy as np

import torch
import torch.autograd as autograd
import torch.utils.data as data
import torchvision
import torch.nn.functional as F
import torchvision.datasets.folder
import torchvision.transforms as transforms

from dataset import Dataset
from res152_wide import get_model as get_model1
from inres import get_model as  get_model2
from v3 import get_model as get_model3
from resnext101 import get_model as get_model4
import time
import sys
from helpers import *

import cv2

npop = 200 # 150 for titanxp     # population size
sigma = 0.1    # noise standard deviation
alpha = 0.02  # learning rate
# alpha = 0.001  # learning rate
boxmin = 0
boxmax = 1
boxplus = (boxmin + boxmax) / 2.
boxmul = (boxmax - boxmin) / 2.

epsi = 0.031

# samples =2

def softmax(x):
    return(np.exp(x)/np.sum(np.exp(x),-1,keepdims=True))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--imagenet-path', type=str, default='../../obfuscated_zoo/imagenet_val',
            help='path to the test_batch file from http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100)


    parser.add_argument('--no-gpu', action='store_true', default=False,
                        help='disables GPU training')
    args = parser.parse_args()


    tf = transforms.Compose([
        transforms.Scale([299, 299]),
        transforms.ToTensor()
    ])

    mean_torch = autograd.Variable(
        torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1]).astype('float32')).cuda(), volatile=True)
    std_torch = autograd.Variable(
        torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1]).astype('float32')).cuda(), volatile=True)
    mean_tf = autograd.Variable(
        torch.from_numpy(np.array([0.5, 0.5, 0.5]).reshape([1, 3, 1, 1]).astype('float32')).cuda(), volatile=True)
    std_tf = autograd.Variable(
        torch.from_numpy(np.array([0.5, 0.5, 0.5]).reshape([1, 3, 1, 1]).astype('float32')).cuda(), volatile=True)

    test_loss = 0
    correct = 0
    total = 0
    totalImages = 0
    succImages = 0
    faillist = []

    # set up TensorFlow session


    # initialize a model

    config, resmodel = get_model1()
    config, inresmodel = get_model2()
    config, incepv3model = get_model3()
    config, rexmodel = get_model4()
    net1 = resmodel.net
    net2 = inresmodel.net
    net3 = incepv3model.net
    net4 = rexmodel.net

    net1 = torch.nn.DataParallel(net1,device_ids=range(torch.cuda.device_count())).cuda()
    net2 = torch.nn.DataParallel(net2,device_ids=range(torch.cuda.device_count())).cuda()
    net3 = torch.nn.DataParallel(net3,device_ids=range(torch.cuda.device_count())).cuda()
    net4 = torch.nn.DataParallel(net4,device_ids=range(torch.cuda.device_count())).cuda()



    checkpoint = torch.load('../all_models/guided-denoiser/denoise_res_015.ckpt')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        resmodel.load_state_dict(checkpoint['state_dict'])
    else:
        resmodel.load_state_dict(checkpoint)

    checkpoint = torch.load('../all_models/guided-denoiser/denoise_inres_014.ckpt')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        inresmodel.load_state_dict(checkpoint['state_dict'])
    else:
        inresmodel.load_state_dict(checkpoint)

    checkpoint = torch.load('../all_models/guided-denoiser/denoise_incepv3_012.ckpt')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        incepv3model.load_state_dict(checkpoint['state_dict'])
    else:
        incepv3model.load_state_dict(checkpoint)

    checkpoint = torch.load('../all_models/guided-denoiser/denoise_rex_001.ckpt')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        rexmodel.load_state_dict(checkpoint['state_dict'])
    else:
        rexmodel.load_state_dict(checkpoint)

    if not args.no_gpu:
        inresmodel = torch.nn.DataParallel(inresmodel,device_ids=range(torch.cuda.device_count())).cuda()
        resmodel = torch.nn.DataParallel(resmodel,device_ids=range(torch.cuda.device_count())).cuda()
        incepv3model = torch.nn.DataParallel(incepv3model,device_ids=range(torch.cuda.device_count())).cuda()
        rexmodel = torch.nn.DataParallel(rexmodel,device_ids=range(torch.cuda.device_count())).cuda()
#
#         inresmodel = inresmodel.cuda()
#         resmodel = resmodel.cuda()
#         incepv3model = incepv3model.cuda()
#         rexmodel = rexmodel.cuda()
    inresmodel.eval()
    resmodel.eval()
    incepv3model.eval()
    rexmodel.eval()


    # initialize a data provider for CIFAR-10 images
    provider = ImageNet(args.imagenet_path, (299,299,3))

    target_list = [10,11,13,23,33,46,51,57,74,77,79,85,98,115,122,125]

    start = 150
    end = 950
    total = 0
    attacktime = 0
    imageno = []
    for i in range(start, end):
#         if i not in target_list:
#             continue
        success = False
        print('evaluating %d of [%d, %d)' % (i, start, end))
        inputs, targets= provider[i]
        modify = np.random.randn(1,3,32,32) * 0.001


        input_var = autograd.Variable(torch.from_numpy(inputs.transpose(2,0,1)).cuda(), volatile=True)
        input_tf = (input_var - mean_tf) / std_tf
        input_torch = (input_var - mean_torch) / std_torch

        logits1 = F.softmax(net1(input_torch,True)[-1],-1)
        logits2 = F.softmax(net2(input_tf,True)[-1],-1)
        logits3 = F.softmax(net3(input_tf,True)[-1],-1)
        logits4 = F.softmax(net4(input_torch,True)[-1],-1)

        logits = ((logits1+logits2+logits3+logits4).data.cpu().numpy())/4


        # print(logits)
        if np.argmax(logits) != targets:
            print('skip the wrong example ', i)
            print('max label {} , target label {}'.format(np.argmax(logits), targets))
            continue
        totalImages += 1
        episode_start =time.time()


        for runstep in range(200):

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


#                 realclipinput = realclipinput.transpose(0, 2, 3, 1)
                realclipinput = np.squeeze(realclipinput)
                realclipinput = np.asarray(realclipinput,dtype = 'float32')


                # realclipinput_expand = []
                # for x in range(samples):
                #     realclipinput_expand.append(realclipinput)
                # realclipinput_expand = np.array(realclipinput_expand)

                input_var = autograd.Variable(torch.from_numpy(realclipinput).cuda(), volatile=True)
                input_tf = (input_var - mean_tf) / std_tf
                input_torch = (input_var - mean_torch) / std_torch

                logits1 = F.softmax(net1(input_torch, True)[-1],-1)
                logits2 = F.softmax(net2(input_tf, True)[-1],-1)
                logits3 = F.softmax(net3(input_tf, True)[-1],-1)
                logits4 = F.softmax(net4(input_torch, True)[-1],-1)

                logits = logits1 + logits2 + logits3 + logits4

                outputsreal = (logits.data.cpu().numpy()[0])/4


                print('probs ',np.sort(outputsreal)[-1:-6:-1])
                print('target label ', np.argsort(outputsreal)[-1:-6:-1])
                print('negative_probs ', np.sort(outputsreal)[0:3:1])
                sys.stdout.flush()
                # print(outputsreal)

                #print(np.abs(realclipdist).max())
                #print('l2real: '+str(l2real.max()))
                # print(outputsreal)
                if (np.argmax(outputsreal) != targets) and (np.abs(realclipdist).max() <= epsi):
                    attacktime += time.time()-episode_start
                    print('episode time : ', time.time()-episode_start)
                    print('atack time : ', attacktime)
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
#             clipinput = clipinput.transpose(0, 2, 3, 1)
            clipinput = np.squeeze(clipinput)
            clipinput = np.asarray(clipinput,dtype = 'float32')
            # clipinput_expand = []
            # for x in range(samples):
            #     clipinput_expand.append(clipinput)
            # clipinput_expand = np.array(clipinput_expand)
            # clipinput_expand = clipinput_expand.reshape((samples * npop, 299, 299, 3))
#             clipinput = clipinput.reshape((npop, 299, 299, 3))

            input_var = autograd.Variable(torch.from_numpy(clipinput).cuda(), volatile=True)
            input_tf = (input_var - mean_tf) / std_tf
            input_torch = (input_var - mean_torch) / std_torch

            logits1 = F.softmax(net1(input_torch, True)[-1],-1)
            logits2 = F.softmax(net2(input_tf, True)[-1],-1)
            logits3 = F.softmax(net3(input_tf, True)[-1],-1)
            logits4 = F.softmax(net4(input_torch, True)[-1],-1)

            logits = logits1 + logits2 + logits3 + logits4

            outputs = (logits.data.cpu().numpy())/4
            
#             print('input_time : ', time.time()-input_start,flush=True)

            target_onehot = target_onehot.repeat(npop,0)

            outputs = np.log(outputs)
            real = (target_onehot * outputs).sum(1)
            other = ((1. - target_onehot) * outputs - target_onehot * 10000.).max(1)[0]

#             real = np.log((target_onehot * outputs).sum(1)+1e-30)
#             other = np.log(((1. - target_onehot) * outputs - target_onehot * 10000.).max(1)[0]+1e-30)

            loss1 = np.clip(real - other, 0.,1000)

            Reward = 0.5 * loss1
#             Reward = l2dist

            Reward = -Reward

            A = (Reward - np.mean(Reward)) / (np.std(Reward)+1e-7)


            modify = modify + (alpha/(npop*sigma)) * ((np.dot(Nsample.reshape(npop,-1).T, A)).reshape(3,32,32))
#             print('one step time : ', time.time()-step_start)
        if not success:
            faillist.append(i)
            print('failed: ',faillist)
#         print('episode time : ', time.time()-episode_start,flush=True)
    print(faillist)
    success_rate = succImages/float(totalImages)




    print('attack success rate: %.2f%% (over %d data points)' % (success_rate*100, args.end-args.start))

if __name__ == '__main__':
    main()
