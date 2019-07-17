#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 12:42:45 2019

@author: wangchong
"""

import numpy as np
from matplotlib import pyplot as plt
from model.InfomaxExplicit import InfoMax;

class Tetanic:
    def __init__(self, trials, network_size, num_to_stim):
        self.net = InfoMax(dim = network_size,
                           GAMMA = 1e-3,
                           BETA = 1,
                           G = 1.5,
                           bias = 0,
                           sparsity = 1);
                           
        self.network_size = network_size;
        self.trials = trials;
        self.num_to_stim = num_to_stim;
        
        self.ItoH = np.random.randn(self.network_size, self.num_to_stim);
        self.sigma = 15;
        self.sign = np.round(np.random.rand(num_to_stim, 1))*2-1;
        self.stimuli = np.random.randn(self.num_to_stim, self.trials)*self.sigma + 0*self.sign;
        
#        self.stimuli = np.concatenate((np.eye(network_size//4), np.flip(np.eye(network_size//4), [0])),1)
#        self.stimuli = np.eye(network_size)[:, 0:network_size/2];
#        self.stimuli = (np.eye(self.num_to_stim) + np.flip(np.eye(self.num_to_stim), [0]))*10-5;
#        self.stimuli = np.round(np.random.rand(np.round(num_to_stim),np.round(num_to_stim))*0.55)*10-5
#        self.stimuli = 50*np.eye(self.num_to_stim)[np.random.permutation(self.num_to_stim)].T-25;
        
    def stimulate(self):
        
        global dw, recording;
        dw = np.zeros(self.trials);
        recording = np.zeros((self.network_size, self.trials));
        
        for i in range(self.trials):
            if (i%200==0):
                print(i);
#            if i>=int(self.trials/2):
#                self.net.gamma = 0;
#                total_input = 0;
            total_input = np.matmul(self.ItoH, self.stimuli[:, i%self.num_to_stim].reshape(-1, 1));
            
            recording[:, i], dw[i] = self.net.trainStep(total_input);
            
            
# =============================================================================
#             if i<self.trials/2:
#                 total_input = np.concatenate((self.stimuli[:, i%self.num_to_stim], np.zeros((self.network_size-self.num_to_stim)))).reshape(-1, 1);
#             else:
#                 total_input = 0;
#             recording[:, i], dw[i] = self.net.trainStep(total_input);
#             
#             if i==self.trials/4:
#                 self.net.gamma /= 10;
#             elif i==self.trials/2:
#                 self.net.gamma *= 10;
#             elif i==self.trials*3/4:
#                 self.net.gamma /= 10;
# =============================================================================
        
        fig, (ax1, ax2) = plt.subplots(2);
        ax1.plot(dw);
        ax2.imshow(recording);
        ax2.set_aspect("auto");
        
        fig, (ax3, ax4) = plt.subplots(1, 2);
        ax3.imshow(self.net.w, cmap="seismic");
        ax4.hist(self.net.w.flatten(), bins=20);
        
        return self.net.w;
        
        
if __name__ == "__main__":
    test = Tetanic(10000, 128, 16);
    w = test.stimulate();
            