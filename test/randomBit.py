#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 20:48:41 2019

@author: wangchong
"""
import numpy as np
from matplotlib import pyplot as plt
from model.InfomaxBCM import InfoMax


class Tetanic():
    def __init__(self, network_size):
        self.network_size = network_size
        self.net = InfoMax(dim = network_size, 
                           GAMMA = 1e-4, 
                           BETA = network_size, 
                           SIGMA = 2,
                           G = 2,
                           bias = 1, 
                           sparsity = 0.1);
        
#        self.stimuli = np.concatenate((np.eye(network_size//4), np.flip(np.eye(network_size//4), [0])),1)
#        self.stimuli = np.eye(network_size)[:, 0:network_size/2];
        self.stimuli = np.eye(network_size) + np.flip(np.eye(network_size), [0])
#        self.stimuli = np.round(np.random.rand(np.round(network_size),np.round(network_size))*0.55)
        
    def stimulate(self, trials):
        stim_shape = self.stimuli.shape;
        trainActivity = np.zeros((self.network_size, trials))
        testActivity = np.zeros((self.network_size, trials))
        
        for i in range(trials):
            print (i);
            
            total_input = np.concatenate((50*self.stimuli[:, i%stim_shape[1]].reshape(stim_shape[0], 1)-25,\
                            np.zeros((self.network_size-stim_shape[0], 1))),axis=0); 
            trainActivity[:, i] = self.net.trainStep(total_input)

        for i in range(trials):
            print (i);
            
            total_input = 0; 
            testActivity[:, i] = self.net.trainStep(total_input)

        plt.imshow(testActivity);
        plt.axes().set_aspect('auto')

#        plt.hist(self.net.w.flatten(),bins=2000);
        
if __name__== "__main__":
    test = Tetanic(526);
    test.stimulate(3000);    