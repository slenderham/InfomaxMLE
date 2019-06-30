#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import numpy as np
from scipy.special import expit
from scipy.special import softmax

class RNN:
    def __init__(self, inputDim, recDim, outDim):
        
        p = 0.1; # sparsity param
        g = 1.5; # variance of reservoir weight
        
        self.recDim = recDim;
        self.inputDim = inputDim;
        self.outDim = outDim;
        
        # learning rate
        self.rIH = 3e-4
        self.rHH = 3e-4
        self.rHO = 3e-4
        
        # inverse of time constant for membrane voltage
#        self.tau_v = np.clip(0.1+np.random.randn(self.recDim,1)*0.01, 0.05, 0.15);
        self.tau_v = 0.1;
        
        # inverse time constant for calculating average reward
        self.tau_r = 0.01;
        
        # inverse time constant for output smoothing
        self.kappa = 1;
        
        # inverse temperature for the sigmoid
        self.beta = 17;
        
        # refractory variable (necessary?)
        self.gamma = 0;
        
        # weight decay
        self.lmbda = 0;

        # hidden and readout state initialization
        self.h = np.zeros((recDim, 1));
        self.o = np.zeros((outDim, 1));
        
        # membrane voltage
#        self.v = np.random.randn(recDim, 1);
        self.v = np.zeros((recDim, 1));
        
        # initialize weights
        self.IH = (2*np.random.rand(recDim, inputDim+1)-1);
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
        
        # moving average of error 
        self.rBar = -np.log(inputDim);
        
        # bias input
        self.b = 0;
        
    def trainStep(self, instr, target):
                
        # integrate input
        instr_aug = np.concatenate((np.ones((1, 1)), instr), axis=0);
        dvt = np.matmul(self.IH, instr_aug) + np.matmul(self.HH, self.h) + self.b;
        
        self.v = (1-self.tau_v)*self.v + self.tau_v*dvt;
        
        # sample with logistic distribution = diff of gumbel RV
        noise = np.random.logistic(0, 1, size=self.h.shape);
        
        # spike or not
        prob = expit(self.beta*(self.v+noise));
        new_states = np.round(prob);
        
        # derivative of log prob w.r.t membrane potential
        dv = (1-new_states)*prob + new_states*(1-prob)
        
        # output and error
        self.o = softmax(np.matmul(self.HO, new_states));
        er = (self.o-target);
        
        # filter the input to readout based on kappa
        self.eHO = (1-self.kappa)*self.eHO + self.kappa*new_states.T;
        
        # update HO
        dHO = np.outer(er, self.eHO.T);
        self.HO -= self.rHO*dHO + self.lmbda*self.HO;
        
        # reward
        r = np.log(np.matmul(self.o.T, target));
                
        # calculate gradient and update eligibility trace
        
        dHH = (r-self.rBar)*np.outer(dv, self.eHH);
        dIH = (r-self.rBar)*np.outer(dv, self.eIH);
        
        self.eHH = self.tau_v*self.h.T + (1-self.tau_v)*self.eHH
        self.eIH = self.tau_v*instr_aug.T + (1-self.tau_v)*self.eIH
                
        self.HH += self.rHH*dHH + self.lmbda*self.HH;
        self.IH += self.rIH*dIH + self.lmbda*self.IH;
        
        # set diagonal elem of HH to 0
        self.HH -= self.HH*np.eye(self.recDim) + self.gamma*np.eye(self.recDim);
        
        self.h = new_states;
        
        # filter the input to readout based on kappa
        self.rBar = (1-self.tau_r)*self.rBar + self.tau_r*r;
        
        # get output onehot
        out = np.zeros(self.outDim);
        out[np.argmax(self.o)] = 1;
        
        return out, self.h.squeeze(), np.linalg.norm(dHH);
    
    def testStep(self, instr):
        # integrate input
        instr_aug = np.concatenate((np.ones((1, 1)), instr), axis=0);
        dv = np.matmul(self.IH, instr_aug) + np.matmul(self.HH, self.h);
        
        self.v = (1-self.tau_v)*self.v + self.tau_v*dv;
        
#        # sample with logistic distribution = diff of gumbel RV
#        noise = np.random.logistic(0, 1, size=self.h.shape);
        noise = 0;
        # spike or not
        prob = expit(self.beta*(self.v+noise));
        new_states = np.round(prob);
        
        # output and error
        self.o = (1-self.kappa)*self.o + self.kappa*np.matmul(self.HO, new_states);
        
        self.h = new_states;
        
        # get output onehot
        out = np.zeros(self.outDim);
        out[np.argmax(self.o)] = 1;
        
        return out, self.h.squeeze();
        