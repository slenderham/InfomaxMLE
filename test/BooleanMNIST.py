#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 21:46:40 2019

@author: wangchong
"""

import numpy as np
from torchvision import datasets, transforms
import torch
from model.ET_MI_categorical_explicit import RNN

def loadCertainDigits(digit, toTrain):
    toFilter = datasets.MNIST(root="./data", train=toTrain, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]));
    
    idx = toFilter.targets==digit;
    toFilter.data = toFilter.data[idx];
    toFilter.targets = toFilter.targets[idx];
    
    return toFilter;

class BooleanAlgebra():

    def __init__(self):
        self.opToRun = int(input("Which operation? (0:AND, 1:OR, 2:XOR) "));
        self.trainTrial = int(input("How many training trials? "));
        self.testTrial = int(input("How many test trials? "));

        inputDim = 784;
        recDim = 128;
        outDim = 2;
        
        # input, smoothed +-1
        self.bit = [5*smooth(np.array([[-1]]), width = width), 
                    5*smooth(np.array([[1]]), width = width)];

        
        global trainRecordingX, trainRecordingR, testRecordingX, testRecordingR;
        trainRecordingX = np.zeros((recDim, self.trainTrial*3*self.dur));
        testRecordingX = np.zeros((recDim, self.testTrial*3*self.dur));
        
        trainRecordingR = np.zeros((recDim, self.trainTrial*3*self.dur));
        testRecordingR = np.zeros((recDim, self.testTrial*3*self.dur));
        
        global trainOut, trainTarget, testOut, testTarget;
        trainOut = np.zeros(self.trainTrial*3*self.dur);
        trainTarget = np.zeros(self.trainTrial*3*self.dur);
        testOut = np.zeros(self.testTrial*3*self.dur);
        testTarget = np.zeros(self.testTrial*3*self.dur);
        
        global w_dot, k_dot;
        w_dot = np.zeros((self.trainTrial*3*self.dur))
        k_dot = np.zeros((self.trainTrial*3*self.dur))
        
        self.turn = True;
        
        # how many iterations until plot
        self.n_plot = 5;
        global ss, gg;
        ss = 0;
        gg = self.n_plot*self.dur*3;        
        
        self.net = RNN(inputDim, recDim, )
        
    def experiment(self):
        # training phase: 
        
        orderOfTask = ["AND", "OR", "XOR"];
        operation = [(lambda x,y: x and y),
                     (lambda x,y: x or y),
                     (lambda x,y: x  ^ y)];
        
        print("Running "+orderOfTask[self.opToRun]);

        # training
        for i in range(self.trainTrial):
            print(i);
            # first bit
            bitOne = int(np.round(np.random.rand()));
            smoothedBitOne = self.bit[bitOne];
            for j in range(self.dur):
                trainOut[3*i*self.dur+j], \
                trainRecordingX[:, 3*i*self.dur+j], \
                trainRecordingR[:, 3*i*self.dur+j], \
                w_dot[3*i*self.dur+j], \
                k_dot[3*i*self.dur+j] \
                = self.net.trainStep(smoothedBitOne[j:j+1, :], np.array([[0]]));
                
                trainTarget[3*i*self.dur+j] = 0
                
            bitTwo = int(np.round(np.random.rand()));
            smoothedBitTwo = self.bit[bitTwo];
            for j in range(self.dur):
                trainOut[(3*i+1)*self.dur+j], \
                trainRecordingX[:, (3*i+1)*self.dur+j], \
                trainRecordingR[:, (3*i+1)*self.dur+j], \
                w_dot[(3*i+1)*self.dur+j], \
                k_dot[(3*i+1)*self.dur+j] \
                = self.net.trainStep(smoothedBitTwo[j:j+1, :], np.array([[0]]));
                
                trainTarget[(3*i+1)*self.dur+j] = 0;
                
            bitTarget = operation[self.opToRun](bitOne, bitTwo);
            smoothedBitTarget = self.bit[bitTarget];
            for j in range(self.responseDur):
                trainOut[(3*i+2)*self.dur+j], \
                trainRecordingX[:, (3*i+2)*self.dur+j], \
                trainRecordingR[:, (3*i+2)*self.dur+j], \
                w_dot[(3*i+2)*self.dur+j], \
                k_dot[(3*i+2)*self.dur+j] \
                = self.net.trainStep(np.array([[0]]), 2*bitTarget-1);
#                = self.net.trainStep(np.array([[0]]), smoothedBitTarget[j:j+1]);
                

        
#                trainTarget[(3*i+2)*self.dur+j] = smoothedBitTarget[j:j+1];
                trainTarget[(3*i+2)*self.dur+j] = 2*bitTarget-1;
                
            if (i+1)%self.n_plot == 0:
                global ss, gg;
                draw_fig();
                ss = gg;
                gg += self.n_plot*3*self.dur;
        # testing
        for i in range(self.testTrial):
            print(i);
            # first bit
            bitOne = int(np.round(np.random.rand()));
            smoothedBitOne = self.bit[bitOne];
            for j in range(self.dur):
                testOut[3*i*self.dur+j], \
                testRecordingX[:, 3*i*self.dur+j], \
                testRecordingR[:, 3*i*self.dur+j] \
                = self.net.testStep(smoothedBitOne[j:j+1, :]);
                
                testTarget[3*i*self.dur+j] = 0;
                
            bitTwo = int(np.round(np.random.rand()));
            smoothedBitTwo = self.bit[bitTwo];
            for j in range(self.dur):
                testOut[(3*i+1)*self.dur+j], \
                testRecordingX[:, (3*i+1)*self.dur+j], \
                testRecordingR[:, (3*i+1)*self.dur+j] \
                = self.net.testStep(smoothedBitTwo[j:j+1, :]);
                
                testTarget[(3*i+1)*self.dur+j] = 0;
                
            bitTarget = operation[self.opToRun](bitOne, bitTwo);
            smoothedBitTarget = self.bit[bitTarget];
            for j in range(self.responseDur):
                testOut[(3*i+2)*self.dur+j], \
                testRecordingX[:, (3*i+2)*self.dur+j], \
                testRecordingR[:, (3*i+2)*self.dur+j] \
                = self.net.testStep(np.array([[0]]));
                
                testTarget[(3*i+2)*self.dur+j] = smoothedBitTarget[j:j+1];
                
        
        # plot connectivity matrix before and after learning
        figEig, (axEig1, axEig2) = plt.subplots(2);
        eigPre = np.linalg.eigvals(np.array(self.net.J_G));
        axEig1.scatter(eigPre.real, eigPre.imag, s=20);
        eigPost = np.linalg.eigvals(np.array(self.net.J_G)+np.outer(self.net.J_z, self.net.w));
        axEig2.scatter(eigPost.real, eigPost.imag, s=20);
        
        figRec, (axRec1, axRec2, axRec3) = plt.subplots(3);
        axRec1.imshow((trainRecordingX.T));
        axRec1.set_title("Training");
        axRec1.set_aspect("auto");
        axRec2.imshow((testRecordingX.T));
        axRec2.set_title("Testing");
        axRec2.set_aspect("auto");
        axRec3.plot(np.array(testOut));
        axRec3.plot(np.array(testTarget));
        axRec3.set_title("Test performance");
        
        
        print(np.sum(np.abs(testOut-testTarget))/self.testTrial);
        
        try:
            return self.net.w, self.net.k, self.net.J_G;
        except:
            return self.net.w, None, self.net.J_G;
    
if __name__=="__main__":
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1);
    test = BooleanAlgebra();
    w, k, J_G = test.experiment();