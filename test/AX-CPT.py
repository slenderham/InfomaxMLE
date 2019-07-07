#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 17:41:29 2019

@author: wangchong
"""

import numpy as np
from smooth import smooth
from matplotlib import pyplot as plt
from model.ET_MI_binary_choice import RNN
from util.draw import draw_fig

class AXCPT:
    def __init__(self):
        self.trainTrial = int(input("How many training trials? "));
        self.testTrial = int(input("How many test trials? "));

        inputDim = 5;
        recDim = 128;
        width = 10;
        self.dur = width*2+1;
        
        self.net = RNN(inputDim, recDim, 1);
        
        global trainTime;
        global testTime;
        trainTime = np.arange(0, self.trainTrial*3*self.dur, self.net.dt);
        testTime = np.arange(0, self.trainTrial*3*self.dur, self.net.dt);

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
        
        self.trialIn = {
                "A": smooth(np.array([[0, 1, 0, 0, 0]]).T, width = width),
                "B": smooth(np.array([[0, 0, 1, 0, 0]]).T, width = width),
                "X": smooth(np.array([[0, 0, 0, 1, 0]]).T, width = width),
                "Y": smooth(np.array([[0, 0, 0, 0, 1]]).T, width = width),
                }
        
        self.respond = smooth(np.array([[1, 0, 0, 0, 0]]).T, width = width);
        self.fixate = smooth(np.zeros((1, inputDim)).T, width = width);
        
        self.target = smooth(np.array([[1]]), width = width);
        
        
        global trainRecordingX, trainRecordingR, testRecordingX, testRecordingR;
        trainRecordingX = np.zeros((recDim, self.trainTrial*3*self.dur));
        testRecordingX = np.zeros((recDim, self.testTrial*3*self.dur));
        
        trainRecordingR = np.zeros((recDim, self.trainTrial*3*self.dur));
        testRecordingR = np.zeros((recDim, self.testTrial*3*self.dur));
        
        self.turn = True;
        
        global w_dot, k_dot;
        w_dot = np.zeros((self.trainTrial*3*self.dur))
        k_dot = np.zeros((self.trainTrial*3*self.dur))
        
        # how many iterations until plot
        self.n_plot = 1;
        global ss, gg;
        ss = 0;
        gg = self.n_plot*self.dur*3;        
        
    def experiment(self):
        # training phase
        global trainOut, trainTarget
        trainOut = np.zeros(self.trainTrial*3*self.dur);
        trainTarget = np.zeros(self.trainTrial*3*self.dur);
        
        for i in range(self.trainTrial):
            currIn = np.random.choice(list(self.trialType.keys()), p=list(self.trialType.values()));
        
            # get input coding cue and probe            
            encodedCue = self.trialIn[currIn[0]]+self.fixate;
            encodedProbe = self.trialIn[currIn[1]]+self.fixate;
            
            for j in range(self.dur):
                
                trainOut[3*i*self.dur+j], \
                trainRecordingX[:, 3*i*self.dur+j], \
                trainRecordingR[:, 3*i*self.dur+j], \
                w_dot[3*i*self.dur+j], \
                k_dot[3*i*self.dur+j] = self.net.trainStep(encodedCue[j:j+1, :], np.array([[0]]));
                
                trainTarget[3*i*self.dur+j] = 0;
            
            for j in range(self.dur):
                
                trainOut[(3*i+1)*self.dur+j], \
                trainRecordingX[:, (3*i+1)*self.dur+j], \
                trainRecordingR[:, (3*i+1)*self.dur+j], \
                w_dot[(3*i+1)*self.dur+j], \
                k_dot[(3*i+1)*self.dur+j] = self.net.trainStep(encodedProbe[j:j+1, :], np.array([[0]]));
                
                trainTarget[(3*i+1)*self.dur+j] = 0;
            
            if (currIn == "AX"):
                for j in range(self.dur):
                    trainOut[(3*i+2)*self.dur+j], \
                    trainRecordingX[:, (3*i+2)*self.dur+j], \
                    trainRecordingR[:, (3*i+2)*self.dur+j], \
                    w_dot[(3*i+2)*self.dur+j], \
                    k_dot[(3*i+2)*self.dur+j] = self.net.trainStep(self.respond[j:j+1, :], self.target[j, :]);
                    
                    trainTarget[(3*i+2)*self.dur+j] = self.target[j:j+1, :];
    #                print(i, trainOut[3*i+2], 1.0)
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
        
                
    
            
            
                                      
            