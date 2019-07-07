# -*- coding: utf-8 -*-

import numpy as np;
import pandas as pd;
from matplotlib import pyplot as plt;
from model.ET_categorical import RNN;

# =============================================================================
# This test hasn't been run.
# =============================================================================

class NameGen():
    def __init__(self, res_size):
        self.res_size = res_size;
        self.io_size = 27;
        self.net = RNN(27, 512, 27);
        
        stim_csv = pd.read_csv("../data/name_matrix.csv", header=None);
        self.stimuli = np.array(stim_csv.values);

        self.alphabet = 'abcdefghijklmnopqrstuvwxyz '
        

    def train(self, trainTrials, testTrials):
#        total number of elements
        N = self.stimuli.shape[1];
        
#        record the activity and weight update
        global trainOut, testOut, trainRecording, testRecording;
        trainRecording = np.zeros((self.res_size, N*trainTrials));
        trainOut = np.zeros((self.io_size, N*trainTrials));
        testRecording = np.zeros((self.res_size, N*trainTrials));
        testOut = np.zeros((self.io_size, N*trainTrials));
        dw = np.zeros(N*trainTrials);
        
        for i in range(trainTrials*trainTrials):
            if(i%5000==0):
                print (i, "/", N*trainTrials)
            
            trainOut[:,i], trainRecording[:,i], dw[i] \
            = self.net.trainStep(self.stimuli[:,i%N], self.stimuli[:,(i+1)%N]);
        
        testOut[:,i], testRecording[:,i] = self.net.testStep(testOut[:, i-1]);
        for i in range(1, testTrials):
            testOut[:,i], testRecording[:,i] = self.net.testStep(testOut[:, i-1]);
            
        fig, (ax1, ax2, ax3) = plt.subplots(3);
        ax1.imshow(self.stimuli);
        testOut[:, 0] = self.stimuli[:, 0];
        
        ax2.imshow(testOut);
        ax2.set_aspect('auto');
        
        ax3.imshow(testRecording);
        ax3.set_aspect('auto');

            
            
        