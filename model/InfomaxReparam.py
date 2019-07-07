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
        self.eHebb = np.zeros((dim, dim));
        self.meanFR = np.zeros((dim, 1));
        
        # time constant for moving average of hebbian product and mean firing rate
        self.tau_e = 0.05;
        self.tau_r = 0.05;
        
    def trainStep(self, ext_in):
        
        # integrate membrane voltage
#        h_aug = np.concatenate(([[1]], self.h));
        h_aug = self.h;
        
        dvt = np.matmul(self.w, h_aug);
        self.v = (1-self.tau_v)*self.v + self.tau_v*dvt;
        
        # noise and spike
        noise = np.random.logistic(0, 1);
        prob_of_spike = expit(self.beta*(self.v - self.b + ext_in));
        soft_step = expit(self.beta*(self.v - self.b + ext_in + noise));
        new_state = np.round(soft_step);
        
        # update eligibility trace
        self.eSpike = (1-self.tau_v)*self.eSpike + self.tau_v*h_aug.T;
        self.meanFR = (1-self.tau_r)*self.meanFR + self.tau_r*prob_of_spike;
        
        # calculate voltage dependent term
# =============================================================================
#         voltage_threshold = np.log((self.meanFR+1e-4) / (1-self.meanFR+1e-4));
#         localVDep = np.outer(soft_step*(1-soft_step)*(self.v - voltage_threshold), self.eSpike)
# =============================================================================
        
        # calculate hebbian term at current time step
        localHebb = np.outer(prob_of_spike*(1-prob_of_spike), self.eSpike);
                             
        # update running average
        self.eHebb = (1-self.tau_e)*self.eHebb + self.tau_e*localHebb;
        
        # calculate final gradient
        dw = np.outer(new_state-prob_of_spike, self.eSpike) \
            - self.eHebb * (new_state-self.meanFR) / (self.meanFR * (1-self.meanFR) + 1e-4);
        
        self.w += self.gamma*(dw);
        self.h = new_state;
        
        return self.h.squeeze(), np.linalg.norm(dw);
        
    
#    def testStep(self, ext_in):
#        x_aug = np.concatenate(([[1]], self.x));
#        vt = np.matmul(self.w, x_aug);
#        
#        prob = expit(vt - self.b + self.sigma*np.random.randn(self.dim,1)+ ext_in);
#        
#        new_state = np.random.binomial(1, prob);
#        self.x = new_state;
#        
#        return prob.squeeze();
        
