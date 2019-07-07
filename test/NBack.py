#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 00:25:52 2019

@author: wangchong
"""
# =============================================================================
# This needs more work
# =============================================================================

import numpy as np
from matplotlib import pyplot as plt
#from model.ET_categorical import RNN
from model.ET_MI_categorical import RNN


class NBack():
    def __init__(self, io_size, network_size, n_back):
        self.network_size = network_size
        
        self.n_back = n_back;
        self.net = RNN(io_size, network_size, io_size);
        self.io_size = io_size;
        
    def stimulate(self, trainTrials, testTrials):        
        
        self.stimuli = np.eye(self.io_size)[np.random.randint(0, self.io_size, size=trainTrials+testTrials)].T;
        
        global trainOut, testOut, trainRecording, testRecording;
        trainOut = np.zeros((self.io_size, trainTrials))
        testOut = np.zeros((self.io_size, testTrials))
        trainRecording = np.zeros((self.network_size, trainTrials));
        testRecording = np.zeros((self.network_size, testTrials));
        
        sumEr = 0;
        sumdW = 0;
        
        for i in range(trainTrials):
            
            trainOut[:, i], trainRecording[:, i], dW \
            = self.net.trainStep(10*self.stimuli[:, i].reshape(-1, 1)-5, \
                                 self.stimuli[:, (i-self.n_back)%trainTrials].reshape(-1, 1));
            
            sumEr += np.sum(np.log(self.net.o)*self.stimuli[:, (i-self.n_back)%trainTrials].reshape(-1, 1));
            sumdW += dW;              
            
            if (i%1000==0 and i!=0):
                
                self.net.rHH *= 0.98;
                self.net.rIH *= 0.98;
                self.net.rHO *= 0.98;
                
                print("\r", i, sumEr/1000, sumdW/(1000*self.network_size**2));
                
                sumEr = 0;
                
        fig, (ax1, ax2, ax3) = plt.subplots(3);
        ax1.imshow(self.stimuli[:, (trainTrials-self.n_back):(trainTrials+testTrials-self.n_back)]);
        ax1.set_aspect('auto');
        
        testEr = 0;
        
        for i in range(testTrials):
            testOut[:, i], testRecording[:, i] \
            = self.net.testStep(10*self.stimuli[:, i+trainTrials].reshape(-1, 1)-5);
            
            testEr += (np.argmax(self.stimuli[:, i+trainTrials-self.n_back]) 
                        == np.argmax(testOut[:, i]));
                
        print(testEr/testTrials);

        ax2.imshow(testOut, cmap="hot");
        ax2.set_aspect('auto');
        
        ax3.imshow(testRecording);
        ax3.set_aspect('auto');

        return self.net.HH;
        
if __name__== "__main__":
    test = NBack(io_size = 8,
                 network_size = 256,
                 n_back = 3);
                 
    w = test.stimulate(trainTrials = 50000,
                       testTrials = 50);