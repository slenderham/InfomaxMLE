# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import softmax
from model.pure_MI import RNN

class ReadOut:
    def __init__(self, recDim, outDim):
        
        self.w_ho = np.zeros(recDim, outDim);
        self.reservoir = RNN(28*28, 64);
        
    def trainStep(self, res, target):
        self.out = softmax(np.matmul(self.w_ho, res));
        self.er = target - out;
        
        