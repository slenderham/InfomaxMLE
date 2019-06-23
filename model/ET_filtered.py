#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 13:17:15 2019

@author: wangchong
"""

import numpy as np
from scipy.special import expit
from scipy.special import softmax

class RNN:
    def __init__(self, inputDim, recDim, outDim):
        
        p = 1; # sparsity param
        g = 10; # variance of reservoir weight
        
        self.recDim = recDim
        self.inputDim = inputDim;
        
        # learning rate
        self.rIH = 2e-4
        self.rHH = 2e-4
        self.rHO = 2e-4
        
        # inverse of time constant for membrane voltage
        self.tau_v = 0.1;
        
        # inverse temperature for the sigmoid
        self.beta = 15;
        
        # refractory variable (necessary?)
        self.gamma = 0;
        
        # weight decay
        self.lmbda = 0;

        # hidden and readout state initialization
        self.h = np.zeros((recDim, 1));
        self.o = np.zeros((outDim, 1));
        
        # readout intgration constant
        self.kappa = 1;
        
        # membrane voltage
        self.v = np.random.randn(recDim, 1);
#        self.v = np.zeros((recDim, 1));
        
        # initialize weights
        self.IH = (2*np.random.rand(recDim, inputDim+1)-1)/np.sqrt(inputDim);
        self.HH = g*np.random.randn(recDim, recDim)/np.sqrt(recDim*p);
        self.HO = (2*np.random.rand(outDim, recDim)-1)/np.sqrt(recDim);
        
        # create sparse HH, have all diag as 0
        mask = np.random.binomial(1, p, size=(recDim, recDim))
        self.HH *= mask;
        self.HH -= self.HH*np.eye(recDim) + self.gamma*recDim*np.eye(recDim);
        
        # eligibility trace
        self.eHH = np.zeros((1, recDim));
        self.eIH = np.zeros((1, inputDim+1));
        self.eHO = np.zeros((1, recDim));
        
    def trainStep(self, instr, target):
        
        # integrate input
        instr_aug = np.concatenate((np.ones((instr.shape[0], 1)), instr), axis=0);
        dv = np.matmul(self.IH, instr_aug) + np.matmul(self.HH, self.h);
        
        self.v = (1-self.tau_v)*self.v + self.tau_v*dv;
        
        # sample with logistic distribution = diff of gumbel RV
        noise = np.random.logistic(0, 1, size=self.h.shape);
#        noise = 0;
        
        # spike or not
        prob = expit(self.beta*(self.v+noise));
        new_states = np.round(prob);
        
        # output and error
        self.o = (1-self.kappa)*self.o + self.kappa*np.matmul(self.HO, new_states);
        er = self.o-target;
        
        # filter the input to readout based on kappa
        self.eHO = (1-self.kappa)*self.eHO + self.kappa*new_states.T;
        
        # update HO
        dHO = np.outer(er, self.eHO.T);
        self.HO -= self.rHO*dHO + self.lmbda*self.HO;
        
        # calculate backprop gradient
        dh = np.matmul(self.HO.T, er);
        dv = dh*(prob)*(1-prob);
        
        # calculate jacobian and update eligibility trace
        self.eHH = self.tau_v*self.h.T + (1-self.tau_v)*self.eHH
        self.eIH = self.tau_v*instr_aug.T + (1-self.tau_v)*self.eIH
        
        dHH = np.outer(dv, self.eHH);
        dIH = np.outer(dv, self.eIH);
        
        self.HH -= self.rHH*dHH + self.lmbda*self.HH;
        self.IH -= self.rIH*dIH + self.lmbda*self.IH;
        
        # set diagonal elem of HH to 0
        self.HH -= self.HH*np.eye(self.recDim) + self.gamma*np.eye(self.recDim);
        
        self.h = new_states;
        
        return self.o.squeeze(), self.v.squeeze(), np.linalg.norm(dHH);
    
    def testStep(self, instr):
        # integrate input
        instr_aug = np.concatenate((np.ones((instr.shape[0], 1)), instr), axis=1).T;
        dv = np.matmul(self.IH, instr_aug) + np.matmul(self.HH, self.h);
        
        self.v = (1-self.tau_v)*self.v + self.tau_v*dv;
        
#        # sample with logistic distribution = diff of gumbel RV
#        noise = np.random.logistic(0, 1, size=self.h.shape);
        # spike or not
        prob = expit(self.beta*(self.v));
        new_states = np.round(prob);
        
        # output and error
        o = np.matmul(self.HO, new_states);
        
        self.h = new_states;
        
        return o.squeeze(), self.v.squeeze();
        