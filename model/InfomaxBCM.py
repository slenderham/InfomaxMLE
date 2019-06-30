#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 22:56:19 2019

@author: wangchong
"""

import numpy as np;
from scipy.special import expit;
from matplotlib import pyplot as plt;

class InfoMax:
    def __init__(self, dim, GAMMA, BETA, SIGMA, G, bias, sparsity):
        
        self.dim = dim;
        
        # binary state
        self.x = np.zeros((dim, 1));
        
        # membrane voltage
        self.v = np.random.randn((dim, 1));
        
        # weights
        self.w = G*np.random.randn(dim, dim+1)/np.sqrt(dim*sparsity);
        mask = np.random.binomial(1, sparsity, size=(dim, dim+1))
        self.w *= mask;
        
        # regularization parameter
        self.beta = BETA;
        
        # learning rate
        self.gamma = GAMMA;
        
        # bias current
        self.b = bias;
        
        # noise parameter
        self.sigma = SIGMA;
        
        # 
        
    def trainStep(self, ext_in):
        
        # integrate membrane voltage
        x_aug = np.concatenate(([[1]], self.x));
        vt = np.matmul(self.w, x_aug);
        
        prob = expit(vt - self.b + self.sigma*np.random.randn(self.dim,1)+ ext_in);
        
        new_state = np.random.binomial(1, prob);
        
        
        self.w += self.gamma*(hebbian - anti_hebbian);
        self.x = new_state;
        
        return prob.squeeze();
    
    def testStep(self, ext_in):
        x_aug = np.concatenate(([[1]], self.x));
        vt = np.matmul(self.w, x_aug);
        
        prob = expit(vt - self.b + self.sigma*np.random.randn(self.dim,1)+ ext_in);
        
        new_state = np.random.binomial(1, prob);
        self.x = new_state;
        
        return prob.squeeze();
        
