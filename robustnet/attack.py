#!/usr/bin/env python3

import argparse
import torch
from torch.autograd import Variable
import torch.optim as optim
import torchvision.transforms as tfs
import torchvision.datasets as dst
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models import *
import numpy as np

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def attack(input_v, label_v, net, c, TARGETED=False):
    n_class = len(classes)
    index = label_v.data.cpu().view(-1,1)
    label_onehot = torch.FloatTensor(input_v.size()[0] , n_class)
    label_onehot.zero_()
    label_onehot.scatter_(1,index,1)
    label_onehot_v = Variable(label_onehot, requires_grad = False).cuda()
	#print(label_onehot.scatter)
    adverse = input_v.data #torch.FloatTensor(input_v.size()).zero_().cuda()
    adverse_v = Variable(adverse, requires_grad=True)
    optimizer = optim.Adam([adverse_v], lr=0.1)
    for _ in range(300):
        optimizer.zero_grad()
        diff = adverse_v - input_v
        output = net(adverse_v)
        real = torch.sum(torch.max(torch.mul(output, label_onehot_v), 1)[0])
        other = torch.sum(torch.max(torch.mul(output, (1-label_onehot_v))-label_onehot_v*10000,1)[0])
        error = c * torch.sum(diff * diff)
        #print(error.size())
        if TARGETED:
            error += torch.clamp(other - real, min=0)
        else:
            error += torch.clamp(real - other, min=0)
        error.backward()
        optimizer.step()
    return adverse_v

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1 and m.affine:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelIn', type=str, default=None)
    parser.add_argument('--c', type=float, default=1.0)
    parser.add_argument('--noise', type=float, default=0)
    opt = parser.parse_args()

    net = VGG("VGG16", opt.noise)
    net = nn.DataParallel(net, device_ids=range(1))
    loss_f = nn.CrossEntropyLoss()
    net.apply(weights_init)
    if opt.modelIn is not None:
        net.load_state_dict(torch.load(opt.modelIn))
    net.cuda()
    loss_f.cuda()
    mean = (0.4914, 0.4822, 0.4465)
    mean_t = torch.FloatTensor(mean).resize_(1, 3, 1, 1).cuda()
    std = (0.2023, 0.1994, 0.2010)
    std_t = torch.FloatTensor(std).resize_(1, 3, 1, 1).cuda()
    transform_train = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    data = dst.CIFAR10("/home/luinx/data/cifar10-py", download=True, train=True, transform=transform_train)
    data_test = dst.CIFAR10("/home/luinx/data/cifar10-py", download=True, train=False, transform=transform_test)
    assert data, data_test
    dataloader = DataLoader(data, batch_size=100, shuffle=True, num_workers=2)
    dataloader_test = DataLoader(data_test, batch_size=100, shuffle=True, num_workers=2)
    count, count2 = 0, 0
    for input, output in dataloader_test:
        input_v, label_v = Variable(input.cuda()), Variable(output.cuda())
        adverse_v = attack(input_v, label_v, net, opt.c)
        _, idx = torch.max(net(input_v), 1)
        _, idx2 = torch.max(net(adverse_v), 1)
        count += torch.sum(label_v.eq(idx)).data[0]
        count2 += torch.sum(label_v.eq(idx2)).data[0]
        print("Count: {}, Count2: {}".format(count, count2))

        adverse_v.data = adverse_v.data * std_t + mean_t
        input_v.data = input_v.data * std_t + mean_t
        adverse_np = adverse_v.cpu().data.numpy().swapaxes(1, 3)
        input_np = input_v.cpu().data.numpy().swapaxes(1, 3)
        plt.subplot(121)
        plt.imshow(np.abs(input_np[0, :, :, :].squeeze()))
        plt.subplot(122)
        plt.imshow(np.abs(adverse_np[0, :, :, :].squeeze()))
        plt.show()

    print("Accuracy: {}, Attach: {}".format(count / len(data), count2 / len(data)))
