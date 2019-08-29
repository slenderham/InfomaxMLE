#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 00:25:52 2019

@author: wangchong
"""

import numpy as np
from matplotlib import pyplot as plt
#from model.ET_categorical import RNN
from model.ET_MI_categorical_explicit_refract import RNN


class NBack():
    def __init__(self, in_size, num_ints, network_size):
        self.network_size = network_size
        
        self.net = RNN(in_size, network_size, 2*num_ints-1);
        self.in_size = in_size;
        self.num_ints = num_ints;
        
    def stimulate(self, trainTrials, testTrials):        
        
        global stimuli, target
        # prepare input
        randInts = np.random.randint(0, self.num_ints, size=trainTrials+testTrials);
        stimuli = np.random.binomial(1, 0.5, size=(self.in_size, self.num_ints))
        
        # prepare target
        sums = (randInts[:-1] + randInts[1:]);
        sums = np.concatenate((np.zeros(1), sums)).astype(int);
        target = np.eye(2*(self.num_ints)-1)[sums].T;
        
        global trainOut, testOut, trainRecording, testRecording, dWs;
        trainOut = np.zeros((trainTrials));
        testOut = np.zeros((testTrials));
        trainRecording = np.zeros((self.network_size, trainTrials));
        testRecording = np.zeros((self.network_size, testTrials));
        dWs = np.zeros((trainTrials));
        
        sumEr = 0;
        
        for i in range(trainTrials):
            
            trainOut[i], trainRecording[:, i], dWs[i] \
            = self.net.trainStep(10*stimuli[:, randInts[i]].reshape(-1, 1)-5 + np.random.randn(self.in_size,1), target[:, i].reshape(-1, 1));
            
            sumEr += np.sum(np.log(self.net.o)*target[:, i].reshape(-1, 1));
            
            if (i%3000==0 and i!=0):
                
                self.net.rHH *= 0.98;
                self.net.rIH *= 0.98;
                self.net.rHO *= 0.98;
                
#                self.net.tau_e*= 0.9;
#                self.net.tau_r *= 0.9;
                
                print("\r", i, sumEr/3000);
                
                sumEr = 0;
                
        fig, (ax1, ax2) = plt.subplots(2);
        ax1.plot(sums[-testTrials:], "go-");
        
        testAcc = 0;
        
        for i in range(testTrials):
            testOut[i], testRecording[:, i] \
            = self.net.testStep(10*stimuli[:, randInts[i+trainTrials]].reshape(-1, 1)-5 + np.random.randn(self.in_size,1));
            
            testAcc += (sums[i+trainTrials] == testOut[i]);
                
        print(testAcc/testTrials);

        ax1.plot(testOut, "bo-");        
        ax2.imshow(testRecording);
        ax2.set_aspect('auto');

        return self.net.IH, self.net.HH;
        
if __name__== "__main__":
    test = NBack(in_size = 100, num_ints = 10, network_size = 64);
    w_ih, w_hh = test.stimulate(trainTrials = 80000, testTrials = 400);