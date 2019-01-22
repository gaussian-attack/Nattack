import robustml
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



npop = 300     # population size
sigma = 0.1    # noise standard deviation
alpha = 0.02  # learning rate
# alpha = 0.001  # learning rate
boxmin = 0
boxmax = 1
boxplus = (boxmin + boxmax) / 2.
boxmul = (boxmax - boxmin) / 2.

epsi = 0.031
epsilon = 1e-30

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
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--noise', type=float, default=0.3)
    parser.add_argument('--modelIn', type=str, default='../all_models/robustnet/noise_0.3.pth')
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


    provider = robustml.provider.CIFAR10(args.cifar_path)
    means = autograd.Variable(
        torch.from_numpy(np.array([0.4914, 0.4822, 0.4465]).reshape([1, 3, 1, 1]).astype('float32')).cuda(), volatile=True)
    #means = torch.tensor(np.array([0.4914,0.4822,0.4465]).reshape(1,3,1,1)).float()
    #stds = torch.tensor(np.array([0.2023, 0.1994, 0.2010]).reshape(1,3,1,1)).float()
    stds = autograd.Variable(
        torch.from_numpy(np.array([0.2023, 0.1994, 0.2010]).reshape([1, 3, 1, 1]).astype('float32')).cuda(), volatile=True)
    #data_test = dst.CIFAR10(args.cifar_path, download=True, train=False, transform=transform_test)
    #dataloader_test = DataLoader(data_test, batch_size=1, shuffle=False, num_workers=1)

    start = 0
    end = 100
    total = 0
    successlist = []
    printlist = []


#     hardlist =  [189, 313, 422, 447, 771, 857, 914, 1078, 1675, 1930, 2091, 2095, 2642, 2987, 3399, 3726, 4375, 4586, 4686, 4868, 5088, 5189, 5245, 5367, 5716, 5806, 6118, 6436, 6481, 6549, 6555, 6574, 6742, 6994, 7347, 7465, 7803, 7835, 8197, 8222, 8236, 8476, 9097, 9507, 9608,9644, 9857, 9872]

    for i in range(start, end):
#         if not i in hardlist:
#             continue
        success = False
        print('evaluating %d of [%d, %d)' % (i, start, end))
        inputs, targets = provider[i]


        input_var = autograd.Variable(torch.from_numpy(inputs.transpose(2, 0, 1)).cuda(), volatile=True)

        #inputs, targets= dataloader_test[i]
        modify = np.random.randn(1,3,32,32) * 0.001
#         print((input_var-means)/stds)

        logits = model((input_var-means)/stds).data.cpu().numpy()

        probs = softmax(logits)
#         print('real logits: ',probs, flush=True)
#         #logits = sess.run(real_logits, feed_dict={input_xs: [inputs]})
#         #print(logits)
#         print('targets: ', targets, flush=True)
        if np.argmax(probs[0]) != targets:
            print('skip the wrong example ', i)
            continue

        totalImages += 1

        for runstep in range(500):
            Nsample = np.random.randn(npop, 3,32,32)# np.random.randn(npop, 3,32,32)

            modify_try = modify.repeat(npop,0) + sigma*Nsample

            newimg = torch_arctanh((inputs-boxplus) / boxmul).transpose(2,0,1)
            #print('newimg', newimg,flush=True)

            inputimg = np.tanh(newimg+modify_try) * boxmul + boxplus
            if runstep % 10 == 0:
                realinputimg = np.tanh(newimg+modify) * boxmul + boxplus
                realdist = realinputimg - (np.tanh(newimg) * boxmul + boxplus)
                realclipdist = np.clip(realdist, -epsi, epsi)
                #print('realclipdist :', realclipdist, flush=True)
                realclipinput = realclipdist + (np.tanh(newimg) * boxmul + boxplus)
                l2real =  np.sum((realclipinput - (np.tanh(newimg) * boxmul + boxplus))**2)**0.5
                #l2real =  np.abs(realclipinput - inputs.numpy())
                print(inputs.shape)
                #outputsreal = model(realclipinput.transpose(0,2,3,1)).data.cpu().numpy()
                input_var = autograd.Variable(torch.from_numpy(realclipinput.astype('float32')).cuda(), volatile=True)

                outputsreal = model((input_var- means)/stds).data.cpu().numpy()[0]
                outputsreal = softmax(outputsreal)
                #print(outputsreal)
                print('probs ', np.sort(outputsreal)[-1:-6:-1])
                print('target label ', np.argsort(outputsreal)[-1:-6:-1])
                print('negative_probs ', np.sort(outputsreal)[0:3:1])

                if (np.argmax(outputsreal) != targets) and (np.abs(realclipdist).max() <= epsi):
                    succImages += 1
                    success = True
                    print('clipimage succImages: '+str(succImages)+'  totalImages: '+str(totalImages))
                    print('lirealsucc: '+str(realclipdist.max()))
                    successlist.append(i)
                    printlist.append(runstep)

#                     imsave(folder+classes[targets[0]]+'_'+str("%06d" % batch_idx)+'.jpg',inputs.transpose(1,2,0))
                    break
            dist = inputimg - (np.tanh(newimg) * boxmul + boxplus)
            clipdist = np.clip(dist, -epsi, epsi)
            clipinput = (clipdist + (np.tanh(newimg) * boxmul + boxplus)).reshape(npop,3,32,32) #.reshape(npop,3,32,32)
            target_onehot =  np.zeros((1,10))


            target_onehot[0][targets]=1.
            clipinput = np.squeeze(clipinput)
            clipinput = np.asarray(clipinput, dtype='float32')
            input_var = autograd.Variable(torch.from_numpy(clipinput).cuda(), volatile=True)
            #outputs = model(clipinput.transpose(0,2,3,1)).data.cpu().numpy()
            outputs = model((input_var-means)/stds).data.cpu().numpy()
            outputs = softmax(outputs)

            target_onehot = target_onehot.repeat(npop,0)



            real = np.log((target_onehot * outputs).sum(1)+epsilon)
            other = np.log(((1. - target_onehot) * outputs - target_onehot * 10000.).max(1)[0]+epsilon)

            loss1 = np.clip(real - other, 0.,1000)

            Reward = 0.5 * loss1
#             Reward = l2dist

            Reward = -Reward

            A = (Reward - np.mean(Reward)) / (np.std(Reward)+1e-7)


            modify = modify + (alpha/(npop*sigma)) * ((np.dot(Nsample.reshape(npop,-1).T, A)).reshape(3,32,32)) #.reshape(3,32,32))
        if not success:
            faillist.append(i)
            print('failed:', faillist)
        else:
            print('successed:',successlist)
    print(faillist)
    success_rate = succImages/float(totalImages)
    print('run steps: ',printlist)
    np.savez('runstep',printlist)
    print('succ rate', success_rate)


#     print('attack success rate: %.2f%% (over %d data points)' % (success_rate*100, args.end-args.start))

if __name__ == '__main__':
    main()
