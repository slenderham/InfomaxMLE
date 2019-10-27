#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 00:25:52 2019

@author: wangchong
"""

import numpy as np
from matplotlib import pyplot as plt
#from model.ET_MI_categorical_explicit_refract import RNN
from model.pure_decoder import Readout as RNN
from torchvision import datasets, transforms
import torch

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

        self.net = RNN(10*10, network_size, 2*io_size-1);

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

        global trainStimuli, trainTarget, testStimuli, testTarget, dW;
        # prepare input
        trainStimuli = np.zeros(trainTrials*trainSetLen);
        trainTarget = np.zeros(trainTrials*trainSetLen);
        testStimuli = np.zeros(testTrials*testSetLen);
        testTarget = np.zeros(testTrials*testSetLen);
        dW = np.zeros(trainTrials*trainSetLen);

        global trainOut, testOut, trainRecording, testRecording;
        trainOut = np.zeros((trainTrials*trainSetLen))
        testOut = np.zeros((testTrials*testSetLen))
        trainRecording = np.zeros((self.network_size, trainTrials*trainSetLen));
        testRecording = np.zeros((self.network_size, testTrials*testSetLen));

        sumEr = 0;
        sumErSq = 0;

        for j in range(trainTrials):
            print("Train Epoch ", j+1);


            for idx, (data, target) in enumerate(self.train_loader):

                trainStimuli[j*trainSetLen + idx] = int(target);
                trainTarget[j*trainSetLen + idx] = int(target) + trainStimuli[j*trainSetLen + idx - 1];

                oneHotTarget = self.oneHot(trainTarget[j*trainSetLen + idx]);

                trainOut[j*trainSetLen + idx], trainRecording[:, j*trainSetLen + idx], dW[j*trainSetLen + idx]\
                = self.net.trainStep(data.numpy().flatten().reshape(-1, 1), oneHotTarget);

                sumEr += np.dot(np.log(self.net.o).T, oneHotTarget);
                sumErSq += np.dot(np.log(self.net.o).T, oneHotTarget)**2;

                if (idx%3000==0 and idx!=0):

                    print(idx, sumEr/3000, (sumErSq/3000-(sumEr/3000)**2));


                    sumEr = 0;
                    sumErSq = 0;

                if (idx%100*j==0):
                    self.net.update();

            self.net.rHH *= 0.6;
            self.net.rIH *= 0.6;
            self.net.rHO *= 0.6;
#            self.net.mi *= 0.9;

            self.net.eHH *= 0;
            self.net.eIH *= 0;

        testAcc = 0;

        for j in range(testTrials):
            print("Test Epoch ", j+1);
            for idx, (data, target) in enumerate(self.test_loader):

                testStimuli[j*testSetLen + idx] = target;
                if (j==0 and idx==0):
                    testTarget[0] = target + trainStimuli[-1];
                else:
                    testTarget[j*testSetLen + idx] = target + testStimuli[j*trainSetLen + idx - 1];

                testOut[j*testSetLen + idx], testRecording[:, j*testSetLen + idx] \
                = self.net.testStep(data.numpy().flatten().reshape(-1, 1));

                testAcc += (testOut[j*testSetLen + idx] == testTarget[j*testSetLen + idx]);

        print(testAcc/(testTrials*testSetLen));

        fig, (ax1, ax2) = plt.subplots(2);
        ax1.plot(testTarget, 'bo-');
        ax1.plot(testOut, 'go-');

        ax2.imshow(testRecording);
        ax2.set_aspect('auto');

        fig, ax3 = plt.subplots(1);
        ax3.plot(dW);

        return self.net.IH, self.net.HH;

    def oneHot(self, target):
        result = np.zeros((self.io_size*2-1, 1));
        result[int(target), 0] = 1;
        return result;

if __name__== "__main__":
    test = NBack(io_size = 10, network_size = 64);

    w_ih, w_hh = test.stimulate(trainTrials = 8, testTrials = 1);