#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 22:56:19 2019

@author: wangchong
"""
# =============================================================================
# terrible idea
# =============================================================================

import numpy as np;
from scipy.special import expit;
from matplotlib import pyplot as plt;

class InfoMax:
    def __init__(self, dim, GAMMA, BETA, G, bias, sparsity):
        
        self.dim = dim;
        
        # binary state
        self.h = np.zeros((dim, 1));
        
        # membrane voltage
        self.v = np.random.randn(dim, 1);
        
        # membrane time constant
        self.tau_v = 0.1;
        
        # weights
        self.w = G*np.random.randn(dim, dim)/np.sqrt(dim*sparsity);
        mask = np.random.binomial(1, sparsity, size=(dim, dim))
        self.w *= mask;
        
        # slope/temperature parameter
        self.beta = BETA;
        
        # learning rate
        self.gamma = GAMMA;
        
        # bias current
        self.b = bias;
        
        # eligibility trace
        self.eSpike = np.zeros((1, dim));
        self.prevFR = np.zeros((dim, 1));
                
    def trainStep(self, ext_in):
        
        # integrate membrane voltage
#        h_aug = np.concatenate(([[1]], self.h));
        h_aug = self.h;
        
        dvt = -self.tau_v + np.matmul(self.w, h_aug);
        self.v += self.tau_v*dvt;
        
        # noise and spike
        noise = np.random.logistic(0, 1);
        prob_of_spike = expit(self.beta*(self.v - self.b + ext_in));
        new_state = np.array(((self.v - self.b + ext_in + noise)>0), dtype=float);
        
        prev_eSpike = self.eSpike;
        self.eSpike = (1-self.tau_v)*self.eSpike + self.tau_v*h_aug.T;
                
        hebbian = np.outer(self.tau_v*dvt*prob_of_spike*(1-prob_of_spike), self.eSpike);
        anti_hebbian = np.outer(prob_of_spike - self.prevFR, prev_eSpike);
        
        self.prevFR = prob_of_spike;
        
        # calculate final gradient
        dw = hebbian - anti_hebbian;
        
        self.w += self.gamma*(dw);
        self.h = new_state;
        
        return self.h.squeeze(), np.linalg.norm(dw)/self.dim**2;
        
