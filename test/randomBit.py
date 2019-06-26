#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 20:48:41 2019

@author: wangchong
"""
import numpy as np
from matplotlib import pyplot as plt
from model.threeFactor_categorical import RNN


class Tetanic():
    def __init__(self, io_size, io_dur, network_size):
        self.network_size = network_size
#        self.net = InfoMax(dim = network_size, 
#                           GAMMA = 1e-4, 
#                           BETA = network_size, 
#                           SIGMA = 2,
#                           G = 2,
#                           bias = 1, 
#                           sparsity = 0.1);
        
        self.net = RNN(io_size, network_size, io_size);
        self.io_size = io_size;
        self.io_dur = io_dur;
        
#        self.stimuli = np.concatenate((np.eye(network_size//4), np.flip(np.eye(network_size//4), [0])),1)
#        self.stimuli = np.eye(network_size)[:, 0:network_size/2];
#        self.stimuli = np.eye(input_size) + np.flip(np.eye(input_size), [0])
#        self.stimuli = np.round(np.random.rand(np.round(io_size),np.round(io_dur))*0.55)
        self.stimuli = np.eye(self.io_size)[np.random.permutation(self.io_size)].T;
        
    def stimulate(self, trainTrials, testTrials):        
        global trainOut, testOut, trainRecording, testRecording;
        trainOut = np.zeros((self.io_size, trainTrials))
        testOut = np.zeros((self.io_size, testTrials))
        trainRecording = np.zeros((self.network_size, trainTrials));
        testRecording = np.zeros((self.network_size, testTrials));
        
        for i in range(trainTrials):
            
            trainOut[:, i], trainRecording[:, i], dW \
            = self.net.trainStep(40*self.stimuli[:, i%self.io_dur].reshape(-1, 1)-20, \
                                 self.stimuli[:, (i+1)%self.io_dur].reshape(-1, 1));
            
            if (i%2000==0 and i!=0):
                print (i, self.net.rBar, dW);
                self.net.rHH *= 0.995;
                self.net.rIH *= 0.995;
                self.net.rHO *= 0.995;

        fig, (ax1, ax2, ax3) = plt.subplots(3);
        ax1.imshow(self.stimuli);
        testOut[:, 0] = self.stimuli[:, 0];
        for i in range(testTrials):
            print (i);
            testOut[:, i], testRecording[:, i] \
            = self.net.testStep(40*testOut[:,(i-1)%self.io_dur].reshape(-1, 1)-20);

        ax2.imshow(testOut);
        ax2.set_aspect('auto');
        
        ax3.imshow(testRecording);
        ax3.set_aspect('auto');

#        plt.hist(self.net.w.flatten(),bins=2000);
        
if __name__== "__main__":
    test = Tetanic(20, 20, 512);
    test.stimulate(55000, 400);