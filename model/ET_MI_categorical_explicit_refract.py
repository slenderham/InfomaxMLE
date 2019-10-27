#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 13:17:15 2019

@author: wangchong
"""

# =============================================================================
# not quite working, need investigation
# =============================================================================

import numpy as np
from scipy.special import expit
from scipy.special import softmax

class RNN:
    def __init__(self, inputDim, recDim, outDim):
        
        p = 0.1; # sparsity param
        g = 1.5; # variance of reservoir weight
        
        self.recDim = recDim
        self.inputDim = inputDim;
        self.outDim = outDim;
        
        # learning rate
        self.rIH = 4e-3
        self.rHH = 4e-3
        self.rHO = 4e-3
        
        # inverse of time constant for membrane voltage
        self.tau_v = np.clip(0.7 + np.random.randn(recDim, 1)*0.1, 0.1, 1);
        
        # inverse of time constant for threshold adaption
        self.tau_t = np.clip(0.9 + np.random.randn(recDim, 1)*0.05, 0.1, 1);
        
        # inverse temperature for the sigmoid
        self.beta = 1;
        
        # refractory variable 
        self.gamma = 10;
        
        # weight decay
        self.lmbda = 0;

        # hidden and readout state initialization
        self.h = np.zeros((recDim, 1));
        self.o = np.zeros((outDim, 1));
        
        # regularization parameter for MI
        self.mi = 0.01;
        
        # membrane voltage
        self.v = np.random.randn(recDim, 1);
        
        # initialize weights
        self.IH = np.random.randn(recDim, inputDim+1)/np.sqrt(inputDim);
        self.HH = g*np.random.randn(recDim, recDim)/np.sqrt(recDim*p);
        self.HO = np.random.rand(outDim, recDim)/np.sqrt(recDim);
        
        # create sparse HH, have all diag as 0
        mask = np.random.binomial(1, p, size=(recDim, recDim))
        self.HH *= mask;
        self.HH -= self.HH*np.eye(recDim);
        
        # eligibility trace
        self.eHH_volt = np.zeros((recDim, recDim));
        self.eIH_volt = np.zeros((recDim, inputDim+1));
        self.eHH_adapt = np.zeros((recDim, recDim));
        self.eIH_adapt = np.zeros((recDim, inputDim+1));

        self.eGradHH = np.zeros((recDim, recDim));
        self.eGradIH = np.zeros((recDim, inputDim+1));
        
        self.meanFR = np.zeros((recDim, 1));
        
        self.thresh = np.zeros((recDim, 1));
        
        # time constant for moving average of hebbian product and mean firing rate
        self.tau_e =  self.tau_r = 0.001;
        
    def trainStep(self, instr, target):
        
        # integrate input
        self.v[self.h==1] = 0;
        instr_aug = np.concatenate((np.ones((1, 1)), instr), axis=0);
        dvt = np.matmul(self.IH, instr_aug) + np.matmul(self.HH, self.h);
        
        self.v = (1-self.tau_v)*self.v + self.tau_v*dvt;
        
        # sample with logistic distribution = diff of gumbel RV
        noise = np.random.logistic(0, 1, size=self.h.shape);
        
        # spike or not
        soft_step = expit(self.beta*(self.v - self.gamma*self.thresh + noise));
        prob = expit(self.beta*(self.v - self.gamma*self.thresh));
        new_states = np.round(soft_step);
        
        # output and error
        self.o = softmax(np.matmul(self.HO, new_states));
        er = self.o-target;
        
        # update HO
        dHO = np.outer(er, new_states);
        self.HO -= self.rHO*dHO + self.lmbda*self.HO;
        
        # calculate backprop gradient
        dh = np.matmul(self.HO.T, er);
        
        # update eligibility traces of voltage
        self.eHH_volt = np.outer(self.tau_v, self.h.T) + (1-self.tau_v)*self.eHH_volt;
        self.eIH_volt = np.outer(self.tau_v, instr_aug.T) + (1-self.tau_v)*self.eIH_volt;
        
        # update moving average of firing rate
        self.meanFR = (1-self.tau_r)*self.meanFR + self.tau_r*prob;
        
        # calculate hebbian term at current time step
        localGradHH = prob * (1-prob) * (self.eHH_volt - self.gamma*self.eHH_adapt);
        localGradIH = prob * (1-prob) * (self.eIH_volt - self.gamma*self.eIH_adapt);
        
        # calculate voltage dependent term
        voltage_threshold = self.v - np.log((self.meanFR) / (1-self.meanFR));
        
        # update running average
        self.eGradHH = (1-self.tau_e)*self.eGradHH + self.tau_e*localGradHH;
        self.eGradIH = (1-self.tau_e)*self.eGradIH + self.tau_e*localGradIH;
        
        hebbianHH = voltage_threshold*localGradHH;
        hebbianIH = voltage_threshold*localGradIH;
        
        antiHebbianHH = self.eGradHH * (prob-self.meanFR) / (self.meanFR*(1-self.meanFR));
        antiHebbianIH = self.eGradIH * (prob-self.meanFR) / (self.meanFR*(1-self.meanFR));
        
        soft_step_prime = soft_step*(1-soft_step);

        dHH = dh*soft_step_prime*(self.eHH_volt - self.gamma*self.eHH_adapt) - self.mi*(hebbianHH - antiHebbianHH);
        dIH = dh*soft_step_prime*(self.eIH_volt - self.gamma*self.eIH_adapt) - self.mi*(hebbianIH - antiHebbianIH);
        
        self.HH -= self.rHH*dHH + self.lmbda*self.HH;
        self.IH -= self.rIH*dIH + self.lmbda*self.IH;
        
        # set diagonal elem of HH to 0
        self.HH -= self.HH*np.eye(self.recDim);
        
        self.h = new_states;
        
        # update eligibility traces of threshold
        pre_factor = self.tau_t*(soft_step)*(1-soft_step);
        self.eIH_adapt = (1-self.tau_t-self.gamma*pre_factor)*self.eIH_adapt + pre_factor*self.eIH_volt;
        self.eHH_adapt = (1-self.tau_t-self.gamma*pre_factor)*self.eHH_adapt + pre_factor*self.eHH_volt;
        
        # reset
        self.v -= self.thresh*self.gamma;
        # update threshold
        self.thresh = (1-self.tau_t)*self.thresh + self.tau_t*new_states;

        
        return np.argmax(self.o), self.v.squeeze(), np.linalg.norm(dHH);
    
    def testStep(self, instr):
        # integrate input
        self.v[self.h==1] = 0;
        instr_aug = np.concatenate((np.ones((1, 1)), instr), axis=0);
        dvt = np.matmul(self.IH, instr_aug) + np.matmul(self.HH, self.h);
        
        self.v = (1-self.tau_v)*self.v + self.tau_v*dvt;
        
#        # sample with logistic distribution = diff of gumbel RV
        noise = np.random.logistic(0, 1, size=self.h.shape);
        # spike or not
        prob = expit(self.beta*(self.v-self.gamma*self.thresh+noise));
        new_states = np.round(prob);
        
        # update threshold
        self.thresh = (1-self.tau_t)*self.thresh + self.tau_t*new_states;
        
        # output and error
        self.o = softmax(np.matmul(self.HO, new_states));
        
        self.h = new_states;
        
        return np.argmax(self.o), self.h.squeeze();
        