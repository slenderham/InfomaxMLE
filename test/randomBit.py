#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 20:48:41 2019

@author: wangchong
"""
import numpy as np
from matplotlib import pyplot as plt
#from model.ET_categorical import RNN
from model.ET_MI_binary import RNN
from util.draw import draw_fig


class RandomBit():
    def __init__(self, io_size, io_dur, network_size):
        self.network_size = network_size
        
        self.net = RNN(io_size, network_size, io_size);
        self.io_size = io_size;
        self.io_dur = io_dur;
        
#        self.stimuli = np.concatenate((np.eye(network_size//4), np.flip(np.eye(network_size//4), [0])),1)
#        self.stimuli = np.eye(network_size)[:, 0:network_size/2];
#        self.stimuli = np.eye(input_size) + np.flip(np.eye(input_size), [0])
        self.stimuli = np.round(np.random.rand(np.round(io_size),np.round(io_dur)))
#        self.stimuli = np.eye(self.io_size)[np.random.permutation(self.io_size)].T;
        
    def stimulate(self, trainTrials, testTrials):        
        global trainOut, testOut, trainRecording, testRecording;
        trainOut = np.zeros((self.io_size, trainTrials*self.io_dur))
        testOut = np.zeros((self.io_size, testTrials*self.io_dur))
        trainRecording = np.zeros((self.network_size, trainTrials*self.io_dur));
        testRecording = np.zeros((self.network_size, testTrials*self.io_dur));
        
        sumEr = 0;
        sumdW = 0;
        
        for i in range(trainTrials*self.io_dur):
            
            trainOut[:, i], trainRecording[:, i], dW \
            = self.net.trainStep(10*self.stimuli[:, i%self.io_dur].reshape(-1, 1)-5, \
                                 self.stimuli[:, (i+1)%self.io_dur].reshape(-1, 1));
            
            sumEr += np.sum(np.log(self.net.o)*self.stimuli[:, (i+1)%self.io_dur].reshape(-1, 1));
            sumdW += dW;              
            
            if (i%2000==0 and i!=0):
                
                self.net.rHH *= 0.999;
                self.net.rIH *= 0.999;
                self.net.rHO *= 0.999;
                
                print("\r", i, sumEr/2000, sumdW/(2000*self.network_size**2));
                
                sumEr = 0;
                
        fig, (ax1, ax2, ax3) = plt.subplots(3);
        ax1.imshow(self.stimuli);
        testOut[:, 0] = self.stimuli[:, 0];
        
        for i in range(1, testTrials*self.io_dur):
            testOut[:, i], testRecording[:, i] \
            = self.net.testStep(10*testOut[:,(i-1)].reshape(-1, 1)-5);

        ax2.imshow(testOut, cmap="hot");
        ax2.set_aspect('auto');
        
        ax3.imshow(testRecording);
        ax3.set_aspect('auto');

        return self.net.HH;
        
if __name__== "__main__":
    test = RandomBit(32, 32, 128);
    w = test.stimulate(1000, 10);