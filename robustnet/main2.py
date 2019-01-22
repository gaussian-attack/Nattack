#!/usr/bin/env python3

import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dst
import torchvision.transforms as tfs
from models import *
from torch.utils.data import DataLoader
import time

def accuracy(dataloader, net):
    data_iter = iter(dataloader)
    count = 0
    total = 0
    for x, y in data_iter:
        vx = Variable(x, volatile=True).cuda()
        tmp = torch.sum(torch.eq(y.cuda(), torch.max(net(vx), dim=1)[1]).data)
        count += int(tmp)
        total += y.size()[0]
    return count / total

def loss(dataloader, net, loss_f):
    data_iter = iter(dataloader)
    total_loss = 0.0
    count = 0
    for x, y in data_iter:
        vx = Variable(x, volatile=True).cuda()
        vy = Variable(y).cuda()
        total_loss += torch.sum(loss_f(net(vx), vy).data)
        count += y.size()[0]
    return total_loss / count

def train_other(dataloader, dataloader_test, net, loss_f, lr, name='adam', max_epoch=10):
    run_time = 0.0
    if name == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=lr)
    elif name == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), lr=lr)
    elif name == 'momsgd':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5.0e-4)
    else:
        print('Not implemented')
        exit(-1)
    for epoch in range(max_epoch):
        beg = time.time()
        data_iter = iter(dataloader)
        for x, y in data_iter:
            vx, vy = Variable(x).cuda(), Variable(y).cuda()
            optimizer.zero_grad()
            lossv = loss_f(net(vx), vy)
            lossv.backward()
            optimizer.step()
        run_time += time.time() - beg
        print("[Epoch {}] Time: {}, Train loss: {}, Train accuracy: {}, Test loss: {}, Test accuracy: {}".format(epoch, run_time, loss(dataloader, net, loss_f), accuracy(dataloader, net), loss(dataloader_test, net, loss_f), accuracy(dataloader_test, net)))


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
    parser.add_argument('--batchSize', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--modelIn', type=str, default=None)
    parser.add_argument('--modelOut', type=str, default=None)
    parser.add_argument('--method', type=str, default="momsgd")
    parser.add_argument('--noise', type=float, default=0.0)
    opt = parser.parse_args()
    print(opt)
    net = VGG("VGG16", opt.noise)
    #net = densenet_cifar()
    #net = GoogLeNet()
    #net = MobileNet(num_classes=100)
    #net = stl10(32)
    net = nn.DataParallel(net, device_ids=range(opt.ngpu))
    #net = Test()
    net.apply(weights_init)
    if opt.modelIn is not None:
        net.load_state_dict(torch.load(opt.modelIn))
    loss_f = nn.CrossEntropyLoss()
    net.cuda()
    loss_f.cuda()
    transform_train = tfs.Compose([
        tfs.RandomCrop(32, padding=4),
        tfs.RandomHorizontalFlip(),
        tfs.ToTensor(),
        tfs.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    data = dst.CIFAR10("../cifar10_data", download=True, train=True, transform=transform_train)
    data_test = dst.CIFAR10("../cifar10_data", download=True, train=False, transform=transform_test)
    assert data, data_test
    dataloader = DataLoader(data, batch_size=opt.batchSize, shuffle=True, num_workers=2)
    dataloader_test = DataLoader(data_test, batch_size=opt.batchSize, shuffle=True, num_workers=2)
    for period in range(opt.epoch // 100):
        train_other(dataloader, dataloader_test, net, loss_f, opt.lr, opt.method, 100)
        opt.lr /= 10
    # save model
    if opt.modelOut is not None:
        torch.save(net.state_dict(), opt.modelOut)

if __name__ == "__main__":
   main()
