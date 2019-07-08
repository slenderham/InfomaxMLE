#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 00:25:52 2019

@author: wangchong
"""

import numpy as np
from matplotlib import pyplot as plt
#from model.ET_categorical import RNN
from model.ET_MI_categorical import RNN


class NBack():
    def __init__(self, io_size, network_size):
        self.network_size = network_size
        
        self.net = RNN(io_size, network_size, 2*io_size-1);
        self.io_size = io_size;
        
    def stimulate(self, trainTrials, testTrials):        
        
        global stimuli, target
        # prepare input
        randInts = np.random.randint(0, self.io_size, size=trainTrials+testTrials);
        stimuli = np.eye(self.io_size)[randInts].T;
        
        # prepare target
        sums = (randInts[:-1] + randInts[1:]);
        sums = np.concatenate((np.zeros(1), sums)).astype(int);
        target = np.eye(2*(self.io_size)-1)[sums].T;
        
        global trainOut, testOut, trainRecording, testRecording;
        trainOut = np.zeros((2*self.io_size-1, trainTrials))
        testOut = np.zeros((2*self.io_size-1, testTrials))
        trainRecording = np.zeros((self.network_size, trainTrials));
        testRecording = np.zeros((self.network_size, testTrials));
        
        sumEr = 0;
        sumdW = 0;
        
        for i in range(trainTrials):
            
            trainOut[:, i], trainRecording[:, i], dW \
            = self.net.trainStep(10*stimuli[:, i].reshape(-1, 1)-5, target[:, i].reshape(-1, 1));
            
            sumEr += np.sum(np.log(self.net.o)*target[:, i].reshape(-1, 1));
            sumdW += dW;              
            
            if (i%1000==0 and i!=0):
                
                self.net.rHH *= 0.98;
                self.net.rIH *= 0.98;
                self.net.rHO *= 0.98;
                
                print("\r", i, sumEr/1000, sumdW/(1000*self.network_size**2));
                
                sumEr = 0;
                
        fig, (ax1, ax2, ax3) = plt.subplots(3);
        ax1.imshow(target[:, -testTrials:]);
        ax1.set_aspect('auto');
        
        testAcc = 0;
        
        for i in range(testTrials):
            testOut[:, i], testRecording[:, i] \
            = self.net.testStep(10*stimuli[:, i+trainTrials].reshape(-1, 1)-5);
            
            testAcc += (sums[i+trainTrials] == np.argmax(testOut[:, i]));
                
        print(testAcc/testTrials);

        ax2.imshow(testOut, cmap="hot");
        ax2.set_aspect('auto');
        
        ax3.imshow(testRecording);
        ax3.set_aspect('auto');

        return self.net.HH;
        
if __name__== "__main__":
    test = NBack(io_size = 6, network_size = 256);
                 
    w = test.stimulate(trainTrials = 40000, testTrials = 100);