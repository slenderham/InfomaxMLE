# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import expit, softmax
import pickle

class Readout:
    def __init__(self, inputDim, recDim, outDim, random=True):

        p = 0.1;
        g = 1.5;

        self.recDim = recDim;

        if random:
            self.IH = np.random.randn(recDim, inputDim+1)/np.sqrt(inputDim);
            self.HH = g*np.random.randn(recDim, recDim)/np.sqrt(recDim*p);
            self.beta = 1;
            self.tau_v = np.clip(0.7 + np.random.randn(self.recDim, 1)*0.1, 0.1, 1);
        else:
            self.IH, self.HH, self.beta, self.tau_v = pickle.load(open("trained_net.txt", "rb"))

        self.rHH = self.rIH = self.rHO = 1e-2

        # hidden and readout state initialization
        self.h = np.zeros((recDim, 1));

        # membrane voltage
        self.v = np.random.randn(recDim, 1);

        # eligibility trace
        self.eHH = np.zeros((recDim, recDim));
        self.eIH = np.zeros((recDim, inputDim+1));

        # readout weights
        self.HO = np.random.rand(outDim, recDim)/np.sqrt(recDim);

        # clip at value
        self.bound = 2;

        self.dHH = np.zeros(self.HH.shape);
        self.dIH = np.zeros(self.IH.shape);
        self.dHO = np.zeros(self.HO.shape);

    def trainStep(self, instr, target):
        # integrate input
        instr_aug = np.concatenate((np.ones((1, 1)), instr), axis=0);
        dvt = np.matmul(self.IH, instr_aug) + np.matmul(self.HH, self.h);

        self.v = (1-self.tau_v)*self.v + self.tau_v*dvt;

#        # sample with logistic distribution = diff of gumbel RV
        noise = np.random.logistic(0, 1, size=self.h.shape);
        # spike or not
        soft_step = expit(self.beta*(self.v+noise));
        new_states = np.round(soft_step);

        # output and error
        self.o = softmax(np.matmul(self.HO, new_states));

        er = self.o-target;

        # update HO
        self.dHO += np.outer(er, new_states.T);

        # calculate backprop gradient
        dh = np.matmul(self.HO.T, er);

        # calculate jacobian and update eligibility trace
        self.eHH = np.outer(self.tau_v, self.h.T) + (1-self.tau_v)*self.eHH
        self.eIH = np.outer(self.tau_v, instr_aug.T) + (1-self.tau_v)*self.eIH

        eHHfromOut = soft_step*(1-soft_step)*self.eHH;
        eIHfromOut = soft_step*(1-soft_step)*self.eIH;

        self.dHH += dh*eHHfromOut;
        self.dIH += dh*eIHfromOut;

        self.h = new_states;

        return np.argmax(self.o), self.v.squeeze(), np.linalg.norm(self.dHH);

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

    def testStep(self, instr):
        # integrate input
        instr_aug = np.concatenate((np.ones((1, 1)), instr), axis=0);
        dvt = np.matmul(self.IH, instr_aug) + np.matmul(self.HH, self.h);

        self.v = (1-self.tau_v)*self.v + self.tau_v*dvt;

#        # sample with logistic distribution = diff of gumbel RV
        noise = np.random.logistic(0, 1, size=self.h.shape);
        # spike or not
        prob = expit(self.beta*(self.v+noise));
        new_states = np.round(prob);

        # output and error
        self.o = softmax(np.matmul(self.HO, new_states));

        self.h = new_states;


        return np.argmax(self.o), self.v.squeeze();