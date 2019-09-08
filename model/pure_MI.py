#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 13:17:15 2019

@author: wangchong
"""

import numpy as np
from scipy.special import expit

class RNN:
    def __init__(self, inputDim, recDim):
        
        p = 0.1; # sparsity param
        g = 1; # variance of reservoir weight
        
        self.recDim = recDim
        self.inputDim = inputDim;
        
        # learning rate
        self.rIH = 1e-3
        self.rHH = 1e-3
        
        # inverse of time constant for membrane voltage
        self.tau_v = np.clip(0.5 + np.random.randn(self.recDim, 1)*0.1, 0.1, 1);
        
        # inverse temperature for the sigmoid
        self.beta = 1;
        
        # refractory variable (necessary?)
        self.gamma = 0;
        
        # desired firing rate and regularization parameter (don't make it close to or bigger than 1)
        self.sigmaBar = 0.1;
        self.lmbda = 0.4;

        # hidden and readout state initialization
        self.h = np.zeros((recDim, 1));
        
        # membrane voltage
        self.v = np.random.randn(recDim, 1);
        
        # initialize weights
        self.IH = (np.random.randn(recDim, inputDim+1))/np.sqrt(inputDim);
        self.HH = g*np.random.randn(recDim, recDim)/np.sqrt(recDim*p);
        
        # create sparse HH, have all diag as 0
        mask = np.random.binomial(1, p, size=(recDim, recDim))
        self.HH *= mask;
        self.HH -= self.HH*np.eye(recDim) + self.gamma*recDim*np.eye(recDim);
        
        mask = np.random.binomial(1, p, size=(recDim, inputDim+1));
        self.IH *= mask;
        
        # eligibility trace
        self.eHH = np.zeros((recDim, recDim));
        self.eIH = np.zeros((recDim, inputDim+1));
        
        self.eGradHH = np.zeros((recDim, recDim));
        self.eGradIH = np.zeros((recDim, inputDim+1));
        
        self.meanFR = np.zeros((recDim, 1));
        
        # time constant for moving average of hebbian product and mean firing rate
        self.tau_e = self.tau_r = 0.01;
        
    def trainStep(self, instr):
        
        # integrate input
        instr_aug = np.concatenate((np.ones((1, 1)), instr), axis=0);
        dvt = np.matmul(self.IH, instr_aug) + np.matmul(self.HH, self.h);
        
        self.v = (1-self.tau_v)*self.v + self.tau_v*dvt;
        
        # sample with logistic distribution = diff of gumbel RV
        noise = np.random.logistic(0, 1, size=self.h.shape);
        
        # spike or not
        soft_step = expit(self.beta*(self.v+noise));
        prob = expit(self.beta*(self.v));
        new_states = np.round(soft_step);
        
        # calculate jacobian and update eligibility trace
        self.eHH = np.outer(self.tau_v, self.h.T) + (1-self.tau_v)*self.eHH
        self.eIH = np.outer(self.tau_v, instr_aug.T) + (1-self.tau_v)*self.eIH
        
        self.meanFR = (1-self.tau_r)*self.meanFR + self.tau_r*prob;
        
        # calculate hebbian term at current time step
        localGradHH = prob*(1-prob)*self.eHH;
        localGradIH = prob*(1-prob)*self.eIH;
        
        # calculate voltage dependent term
        voltage_threshold = self.v - np.log((self.meanFR) / (1-self.meanFR));
        sparse_constraint = self.v - np.log((self.sigmaBar) / (1-self.sigmaBar));
        
        # update running average
        self.eGradHH = (1-self.tau_e)*self.eGradHH + self.tau_e*localGradHH;
        self.eGradIH = (1-self.tau_e)*self.eGradIH + self.tau_e*localGradIH;
        
        hebbianHH = (voltage_threshold - self.lmbda*sparse_constraint)*localGradHH;
        hebbianIH = (voltage_threshold - self.lmbda*sparse_constraint)*localGradIH;
        
        antiHebbianHH = self.eGradHH * ((prob-self.meanFR) / (self.meanFR*(1-self.meanFR)));
        antiHebbianIH = self.eGradIH * ((prob-self.meanFR) / (self.meanFR*(1-self.meanFR)));
        
        dHH = - (hebbianHH - antiHebbianHH);
        dIH = - (hebbianIH - antiHebbianIH);
        
        self.HH -= self.rHH*dHH;
        self.IH -= self.rIH*dIH;
        
        # set diagonal elem of HH to 0
        self.HH -= self.HH*np.eye(self.recDim) + self.gamma*np.eye(self.recDim);
        
        self.h = new_states;
        
        return self.h.squeeze(), np.linalg.norm(dHH), np.linalg.norm(self.HH);
    
    def testStep(self, instr):
        # integrate input
        instr_aug = np.concatenate((np.ones((1, 1)), instr), axis=0);
        dvt = np.matmul(self.IH, instr_aug) + np.matmul(self.HH, self.h);
        
        self.v = (1-self.tau_v)*self.v + self.tau_v*dvt;
        
        # sample with logistic distribution = diff of gumbel RV
        noise = np.random.logistic(0, 1, size=self.h.shape);
        # spike or not
        prob = expit(self.beta*(self.v+noise));
        new_states = np.round(prob);
                
        self.h = new_states;
        
        
        return self.h.squeeze();
        