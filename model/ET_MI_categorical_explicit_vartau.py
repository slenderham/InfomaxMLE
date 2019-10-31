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
		g = 1.5; # variance of reservoir weight

		self.recDim = recDim
		self.inputDim = inputDim;
		self.outDim = outDim;

		# learning rate
		self.rIH = 1e-2
		self.rHH = 1e-2
		self.rHO = 1e-2

		# inverse of time constant for membrane voltage
		self.tau_v = np.clip(0.7 + np.random.rand(self.recDim, 1)*0.1, 0.1, 1);

		# inverse temperature for the sigmoid
		self.beta = 1;

		# refractory variable (necessary?)
		self.gamma = 0;

		# weight decay
		self.lmbda = 0;

		# hidden and readout state initialization
		self.h = np.zeros((recDim, 1));
		self.o = np.zeros((outDim, 1));

		# readout intgration constant
		self.kappa = 1;

		# regularization parameter for MI
		self.mi = 0.1;

		# membrane voltage
		self.v = 0*np.random.randn(recDim, 1);

		# initialize weights
		self.IH = np.random.randn(recDim, inputDim+1)/np.sqrt(inputDim);
		self.HH = g*np.random.randn(recDim, recDim)/np.sqrt(recDim*p);
		self.HO = np.random.rand(outDim, recDim)/np.sqrt(recDim);

		# create sparse HH, have all diag as 0
		mask = np.random.binomial(1, p, size=(recDim, recDim))
		self.HH *= mask;
		self.HH -= self.HH*np.eye(recDim) + self.gamma*recDim*np.eye(recDim);

		# eligibility trace
		self.eHH = np.zeros((recDim, recDim));
		self.eIH = np.zeros((recDim, inputDim+1));
		self.eHO = np.zeros((1, recDim));

		self.eHHfromOut = np.zeros((recDim, recDim));
		self.eIHfromOut = np.zeros((recDim, inputDim+1));

		self.eGradHH = np.zeros((recDim, recDim));
		self.eGradIH = np.zeros((recDim, inputDim+1));

		self.meanFR = expit(np.random.randn(recDim, 1)*1.5);

		# time constant for moving average of hebbian product and mean firing rate
		self.tau_e = self.tau_r = 0.01;

		self.dHH = np.zeros(self.HH.shape);
		self.dIH = np.zeros(self.IH.shape);
		self.dHO = np.zeros(self.HO.shape);

		self.bound = 2;

	def trainStep(self, instr, target):

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

		# output and error
		self.o = softmax((1-self.kappa)*self.o + self.kappa*np.matmul(self.HO, new_states));
		er = self.o-target;

		# filter the input to readout based on kappa
		self.eHO = (1-self.kappa)*self.eHO + self.kappa*new_states.T;

		# update HO
		self.dHO += np.outer(er, self.eHO.T);

		# calculate backprop gradient
		dh = np.matmul(self.HO.T, er);

		# calculate jacobian and update eligibility trace
		self.eHH = np.outer(self.tau_v, self.h.T) + (1-self.tau_v)*self.eHH
		self.eIH = np.outer(self.tau_v, instr_aug.T) + (1-self.tau_v)*self.eIH

		# 0.3*(1-np.abs(self.v+noise))*((np.abs(self.v+noise)<=1))

		self.eHHfromOut = (1-self.kappa)*self.eHHfromOut \
						+ self.kappa*soft_step*(1-soft_step)*((np.abs(self.v+noise)<=5))*self.eHH;
		self.eIHfromOut = (1-self.kappa)*self.eIHfromOut \
						+ self.kappa*soft_step*(1-soft_step)*((np.abs(self.v+noise)<=5))*self.eIH;

		self.meanFR = (1-self.tau_r)*self.meanFR + self.tau_r*prob;

		# calculate hebbian term at current time step
		localGradHH = prob*(1-prob)*self.eHH;
		localGradIH = prob*(1-prob)*self.eIH;

		# calculate voltage dependent term
		voltage_threshold = self.v - np.log((self.meanFR) / (1-self.meanFR));

		# update running average
		self.eGradHH = (1-self.tau_e)*self.eGradHH + self.tau_e*localGradHH;
		self.eGradIH = (1-self.tau_e)*self.eGradIH + self.tau_e*localGradIH;

		hebbianHH = voltage_threshold*localGradHH;
		hebbianIH = voltage_threshold*localGradIH;

		antiHebbianHH = self.eGradHH * (prob-self.meanFR) / (self.meanFR*(1-self.meanFR));
		antiHebbianIH = self.eGradIH * (prob-self.meanFR) / (self.meanFR*(1-self.meanFR));

		self.dHH += dh*self.eHHfromOut - self.mi*(hebbianHH - antiHebbianHH);
		self.dIH += dh*self.eIHfromOut - self.mi*(hebbianIH - antiHebbianIH);


		self.h = new_states;

		return np.argmax(self.o), self.v.squeeze(), np.linalg.norm(self.dHH);

	def testStep(self, instr):
		# integrate input
		instr_aug = np.concatenate((np.ones((1, 1)), instr), axis=0);
		dvt = np.matmul(self.IH, instr_aug) + np.matmul(self.HH, self.h);

		self.v = (1-self.tau_v)*self.v + self.tau_v*dvt;

#		# sample with logistic distribution = diff of gumbel RV
		noise = np.random.logistic(0, 1, size=self.h.shape);
		# spike or not
		prob = expit(self.beta*(self.v+noise));
		new_states = np.round(prob);

		# output and error
		self.o = softmax((1-self.kappa)*self.o + self.kappa*np.matmul(self.HO, new_states));

		self.h = new_states;


		return np.argmax(self.o), self.v.squeeze();

	def update(self):
		self.dHH = np.clip(self.dHH, -self.bound, self.bound);
		self.dIH = np.clip(self.dIH, -self.bound, self.bound);
		self.dHO = np.clip(self.dHO, -self.bound, self.bound);

		self.HO -= self.rHO*self.dHO;
		self.HH -= self.rHH*self.dHH;
		self.IH -= self.rIH*self.dIH;

		# set diagonal elem of HH to 0
		self.HH -= self.HH*np.eye(self.recDim);

		self.dHO *= 0;
		self.dHH *= 0;
		self.dHO *= 0;
