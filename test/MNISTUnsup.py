#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 00:25:52 2019

@author: wangchong
"""

import numpy as np
from matplotlib import pyplot as plt
from model.pure_MI import RNN
from torchvision import datasets, transforms
import torch
import pickle

np.set_printoptions(precision=4)

def loadCertainDigits(io_size, toTrain):
    toFilter = datasets.MNIST(root="./data", train=toTrain, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)),

                       ]), download = True);

    idx = toFilter.targets==0;
    for i in range(1, io_size):
        idx += toFilter.targets==i;

    toFilter.data = toFilter.data[idx];
    toFilter.targets = toFilter.targets[idx];

    return toFilter;

class NBack():
    def __init__(self, io_size, network_size):
        self.network_size = network_size

        self.net = RNN(28*28, network_size);

        # use only digits 0 to io_size-1
        self.io_size = io_size;

        # load data, select only certain subsets
        self.train_loader = torch.utils.data.DataLoader(
                loadCertainDigits(self.io_size, toTrain=True), shuffle=True);

        self.test_loader = torch.utils.data.DataLoader(
                loadCertainDigits(self.io_size, toTrain=False), shuffle=True);

    def stimulate(self, trainTrials, testTrials):

        trainSetLen = len(self.train_loader.dataset);
        testSetLen = len(self.test_loader.dataset);

        global trainStimuli, testStimuli
        # prepare input
        trainStimuli = np.zeros(trainTrials*trainSetLen);
        testStimuli = np.zeros(testTrials*testSetLen);

        global trainRecording, testRecording;
        trainRecording = np.zeros((self.network_size, trainTrials*trainSetLen));
        testRecording = np.zeros((self.network_size, testTrials*testSetLen));

        sumdW = 0;
        sumdWSq = 0;
        sumWnorm = 0;

        for j in range(trainTrials):
            print("Train Epoch ", j+1);
            for idx, (data, target) in enumerate(self.train_loader):

                trainStimuli[j*trainSetLen + idx] = int(target);

                trainRecording[:, j*trainSetLen + idx], dW, wNorm \
                = self.net.trainStep(data.numpy().flatten().reshape(-1, 1));

                sumdW += dW;
                sumWnorm += wNorm;
                sumdWSq += dW**2;

                if (idx%3000==0 and idx!=0):

                    print(idx, sumdW/(3000), (sumdWSq/3000-(sumdW/3000)**2), sumWnorm/3000);

                    sumdW = 0;
                    sumdWSq = 0;
                    sumWnorm = 0;

            self.net.rHH *= 0.8;
            self.net.rIH *= 0.8;
#            self.net.mi *= 0.9;


        for j in range(testTrials):
            print("Test Epoch ", j+1);
            for idx, (data, target) in enumerate(self.test_loader):

                testStimuli[j*testSetLen + idx] = target;

                testRecording[:, j*testSetLen + idx] \
                = self.net.testStep(data.numpy().flatten().reshape(-1, 1));

        fig, (ax1, ax2) = plt.subplots(2);
        ax1.imshow(trainRecording);
        ax1.set_aspect("auto");

        ax2.imshow(testRecording);
        ax2.set_aspect('auto');

        fig, ax = plt.subplots(16,16);
        for i in range(256):
            ax[i//16, i%16].imshow(self.net.IH[i,1:].reshape(28,28),cmap="seismic");

        _ = plt.figure();
        plt.imshow(self.net.HH);


        return self.net.IH, self.net.HH, self.net.beta, self.net.tau_v;

    def oneHot(self, target):
        result = np.zeros((self.io_size*2-1, 1));
        result[int(target), 0] = 1;
        return result;

if __name__== "__main__":
    test = NBack(io_size = 10, network_size = 256);
    net = test.stimulate(trainTrials = 2, testTrials = 1);

    pickle.dump(net, open("trained_net.txt", "wb"))