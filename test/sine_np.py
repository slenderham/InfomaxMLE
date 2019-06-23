#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 22:46:16 2019

@author: wangchong
exit"""

import math
import numpy as np
from matplotlib import pyplot as plt
from model.ET_filtered import RNN

class PatternGen:
    def __init__(self, nsecs):
        dt = 0.1;
        simtime = np.arange(0, nsecs-dt, dt);
        simtime2 = np.arange(1*nsecs, np.floor(1.05*nsecs-dt), dt);
        
        
        amp = 20;
        freq = 1.0/60;
        ft = (amp/1.0)*np.sin(1.0*math.pi*freq*simtime) + \
             (amp/2.0)*np.sin(2.0*math.pi*freq*simtime) + \
             (amp/6.0)*np.sin(3.0*math.pi*freq*simtime) + \
             (amp/3.0)*np.sin(4.0*math.pi*freq*simtime);
#        ft = (amp/1.0)*np.sin(1.0*math.pi*freq*simtime) + \
#             (amp/2.0)*np.sin(2.0*math.pi*freq*simtime);
        self.ft = ft/1.5;
        
        self.clock = amp*np.sin(1.0*math.pi*freq*simtime);
        
        ft2 = (amp/1.0)*np.sin(1.0*math.pi*freq*simtime2) + \
              (amp/2.0)*np.sin(2.0*math.pi*freq*simtime2) + \
              (amp/6.0)*np.sin(3.0*math.pi*freq*simtime2) + \
              (amp/3.0)*np.sin(4.0*math.pi*freq*simtime2);
#        ft2 = (amp/1.0)*np.sin(1.0*math.pi*freq*simtime2) + \
#             (amp/2.0)*np.sin(2.0*math.pi*freq*simtime2);
        self.ft2 = ft2/1.5;
        
        self.net = RNN(1, 128, 1);
        
    def run(self):
        
        trainOut = np.zeros(self.ft.shape);
        trainRecording = np.zeros((self.net.recDim, self.ft.shape[0]));
        
        for i in range(0, list(self.ft.shape)[0]-1):
            trainOut[i], trainRecording[:, i], dW = self.net.trainStep(np.array([[self.clock[i+1]]]), self.ft[i+1]);
           
            if i%1000==0 and i!=0:
                self.net.rHH -= 3e-7;
                self.net.rIH -= 3e-7;
                self.net.rHO -= 3e-7;
                
                self.net.beta *= 1.01;
                
                print(i, trainOut[i]-self.ft[i+1], dW);
        
        testOut = np.zeros(self.ft2.shape);
        testRecording = np.zeros((self.net.recDim, self.ft2.shape[0]));
        for i in range(1, list(self.ft2.shape)[0]):
            testOut[i], testRecording[:, i] = self.net.testStep(np.array([[self.clock[i+1]]]));
            if i%500==0:
                print(i, testOut[i]-self.ft2[i]);
            
        
#        plt.plot(trainOut);
#        plt.plot(self.ft)
#        plt.imshow(self.net.HH);
        
        fig, (ax1, ax2) = plt.subplots(2);
        ax1.plot(testOut);
        ax1.plot(self.ft2);
#        ax1.imshow(trainRecording);
        ax1.set_aspect("auto");
        ax2.imshow(testRecording);
        ax2.set_aspect("auto");
        
#        eigPre = np.linalg.eigvals(self.net.HH);
#        plt.scatter(eigPre.real, eigPre.imag, s=20);
#        eigPost = np.linalg.eigvals(np.array(self.net.J_G+np.outer(self.net.wf.squeeze(), self.net.wo.squeeze())));
#        ax2.scatter(eigPost.real, eigPost.imag, s=20);
if __name__=="__main__":
    test = PatternGen(30000);
    test.run();