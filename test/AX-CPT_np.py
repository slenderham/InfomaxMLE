#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from model.ET_MI_bernoulli_explicit import RNN
from torchvision import datasets, transforms
import torch

def loadCertainDigits(toSelect, toTrain):
    toFilter = datasets.MNIST(root="./data", train=toTrain, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]));
    
    idx = toFilter.targets==toSelect;
        
    toFilter.data = toFilter.data[idx];
    toFilter.targets = toFilter.targets[idx];
    
    return toFilter;

class AXCPT:
    def __init__(self):

        inputDim = 28*28 + 1;
        recDim = 128;
        outDim = 1;
        
        self.net = RNN(inputDim, recDim, outDim);
        
        # input dimensions: 
        #   0: fixation
        #   1, 2: cues, A, B
        #   3, 4: probe: X, Y
        self.trialType = {
                "AX": 0.4,
                "BX": 0.1, 
                "AY": 0.1, 
                "BY": 0.4,
                }
        
        self.trainStim = {
                "A": torch.utils.data.DataLoader(loadCertainDigits(0, toTrain=True), shuffle=True),
                "B": torch.utils.data.DataLoader(loadCertainDigits(1, toTrain=True), shuffle=True),
                "X": torch.utils.data.DataLoader(loadCertainDigits(2, toTrain=True), shuffle=True),
                "Y": torch.utils.data.DataLoader(loadCertainDigits(3, toTrain=True), shuffle=True)
                };
                
        self.testStim = {
                "A": torch.utils.data.DataLoader(loadCertainDigits(0, toTrain=False), shuffle=True),
                "B": torch.utils.data.DataLoader(loadCertainDigits(1, toTrain=False), shuffle=True),
                "X": torch.utils.data.DataLoader(loadCertainDigits(2, toTrain=False), shuffle=True),
                "Y": torch.utils.data.DataLoader(loadCertainDigits(3, toTrain=False), shuffle=True)
                };
        
        # get total number of images
        self.trainTrial = len(self.trainStim["A"].dataset + self.trainStim["B"].dataset \
                              + self.trainStim["X"].dataset + self.trainStim["Y"].dataset);
        self.testTrial = len(self.testStim["A"].dataset + self.testStim["B"].dataset \
                             + self.testStim["X"].dataset + self.testStim["Y"].dataset);
        
        # make iterators for later use
        for loader in self.trainStim:
            loader = iter(loader);
        
        for loader in self.testStim:
            loader = iter(loader);
        
        global trainRecordingX, trainRecordingR, testRecordingX, testRecordingR;
        trainRecordingX = np.zeros((recDim, self.trainTrial));
        testRecordingX = np.zeros((recDim, self.testTrial));
        
        trainRecordingR = np.zeros((recDim, self.trainTrial));
        testRecordingR = np.zeros((recDim, self.testTrial));
                
        global w_dot, k_dot;
        w_dot = np.zeros((self.trainTrial*3*self.dur))
        k_dot = np.zeros((self.trainTrial*3*self.dur))
        
    def experiment(self):
        # training phase
        global trainOut, trainTarget
        trainOut = np.zeros(self.trainTrial*3*self.dur);
        trainTarget = np.zeros(self.trainTrial*3*self.dur);
                
        for i in range(self.trainTrial):
            currIn = np.random.choice(list(self.trialType.keys()), p=list(self.trialType.values()));
            
            image, label = next(self.trainStim[currIn[0]]);
            
            
            if i%5==0:
                print(i)
            else:
                for j in range(self.dur):
                    trainOut[(3*i+2)*self.dur+j], \
                    trainRecordingX[:, (3*i+2)*self.dur+j], \
                    trainRecordingR[:, (3*i+2)*self.dur+j], \
                    w_dot[(3*i+2)*self.dur+j], \
                    k_dot[(3*i+2)*self.dur+j] = self.net.trainStep(self.respond[j:j+1, :], -self.target[j, :]);
                    trainTarget[(3*i+2)*self.dur+j] = -self.target[j:j+1, :];
    #                print(i, trainOut[3*i+2], 1.0)
                if i%5==0:
                    print(i)
                    
            if i%self.n_plot == 0:
                global ss, gg;
                draw_fig();
                ss = gg;
                gg += self.n_plot*3*self.dur;
                        
        print(np.sum(np.abs(trainOut-trainTarget)));
        
        # testing phase
        testOut = np.zeros(self.testTrial*3*self.dur);
        testTarget = np.zeros(self.testTrial*3*self.dur);
#        
        for i in range(self.testTrial):
            currIn = np.random.choice(list(self.trialType.keys()), p=list(self.trialType.values()));

            # get input coding cue and probe            
            encodedCue = self.trialIn[currIn[0]]+self.fixate;
            encodedProbe = self.trialIn[currIn[1]]+self.fixate;
            
            for j in range(self.dur):
                
                testOut[3*i*self.dur+j], \
                testRecordingX[:, 3*i*self.dur+j], \
                testRecordingR[:, 3*i*self.dur+j] = self.net.testStep(encodedCue[j:j+1, :]);
                
                testTarget[3*i*self.dur+j] = 0;
            
            for j in range(self.dur):
                
                testOut[(3*i+1)*self.dur+j], \
                testRecordingX[:, (3*i+1)*self.dur+j], \
                testRecordingR[:, (3*i+1)*self.dur+j] = self.net.testStep(encodedProbe[j:j+1, :]);
                
                testTarget[(3*i+1)*self.dur+j] = 0;
            
            if (currIn == "AX"):                    
                for j in range(self.dur):
                    testOut[(3*i+2)*self.dur+j], \
                    testRecordingX[:, (3*i+2)*self.dur+j], \
                    testRecordingR[:, (3*i+2)*self.dur+j] = self.net.testStep(self.respond[j:j+1, :]);
                    
                    testTarget[(3*i+2)*self.dur+j] = self.target[j:j+1, :];
    #                print(i, trainOut[3*i+2], 1.0)
                if i%5==0:
                    print(i)
            else:
                for j in range(self.dur):
                    testOut[(3*i+2)*self.dur+j], \
                    testRecordingX[:, (3*i+2)*self.dur+j], \
                    testRecordingR[:, (3*i+2)*self.dur+j] = self.net.testStep(self.respond[j:j+1, :]);
                    
                    testTarget[(3*i+2)*self.dur+j] = -self.target[j:j+1, :];
    #                print(i, trainOut[3*i+2], 1.0)
                if i%5==0:
                    print(i)
                
        
#        print(self.net.wo)
        print(1-np.sum(np.abs(testOut-testTarget))/self.testTrial);
        
        # plot connectivity matrix before and after learning
        figEig, (axEig1, axEig2) = plt.subplots(2);
        eigPre = np.linalg.eigvals(np.array(self.net.J_G));
        axEig1.scatter(eigPre.real, eigPre.imag, s=20);
        eigPost = np.linalg.eigvals(np.array(self.net.J_G)+np.outer(self.net.J_z, self.net.w));
        axEig2.scatter(eigPost.real, eigPost.imag, s=20);
        
        figRec, (axRec1, axRec2, axRec3) = plt.subplots(3);
        axRec1.imshow((trainRecordingX.T));
        axRec1.set_title("Training");
        axRec2.imshow((testRecordingX.T));
        axRec2.set_title("Testing");
        axRec3.plot(np.array(testOut));
        axRec3.plot(np.array(testTarget));
        axRec3.set_title("Test performance");
        
        return self.net.w, self.net.k, self.net.J_G;
    
if __name__=="__main__":
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1);
    test = AXCPT();
    w, k, J_G = test.experiment();
        
                
    
            
            
                                      
            