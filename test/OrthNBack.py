#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 00:25:52 2019

@author: wangchong
"""

import numpy as np
import pickle
from matplotlib import pyplot as plt
from model.pure_MI import RNN
from model.pure_decoder import Readout
from model.ET_MI_categorical_explicit_vartau import RNN as Together
from scipy.stats import ortho_group

np.random.seed(0);

class NBack():
	def __init__(self, in_size, num_ints, network_size, whole_obj = False):
		self.network_size = network_size

		self.preNet = RNN(in_size, network_size);

		if (not whole_obj):
			self.net = Readout(in_size, network_size, 4*num_ints-1);
		else:
			self.net = Together(in_size, network_size, 4*num_ints-1);
		self.in_size = in_size;
		self.num_ints = num_ints;

	def stimulate(self, preTrainTrials, trainTrials, testTrials):

		global stimuli, target
		# prepare input
		randInts = np.random.randint(0, self.num_ints, size=trainTrials+testTrials);
		stimuli = ortho_group.rvs(self.in_size)[:,randInts]*self.in_size + 0.1*np.random.randn(self.in_size, trainTrials+testTrials);

		# prepare target
		sums = (randInts[:-3] + randInts[1:-2] + randInts[2:-1] + randInts[3:]);
		sums = np.concatenate((np.zeros(3), sums)).astype(int);
		target = np.eye(4*(self.num_ints)-1)[sums].T;

		global trainOut, testOut, preTrainRecording, trainRecording, testRecording, dWs;
		trainOut = np.zeros((trainTrials));
		testOut = np.zeros((testTrials));
		trainRecording = np.zeros((self.network_size, trainTrials));
		testRecording = np.zeros((self.network_size, testTrials));
		preTrainRecording = np.zeros((self.network_size, preTrainTrials));
		dWs = np.zeros((trainTrials));

		sumdW = 0;
		sumdWSq = 0;
		sumWnorm = 0;

		if preTrainTrials==0:
			idx = 0;
		else:
			idx = np.random.randint(preTrainTrials);
#		idx = 0;
		for i in range(preTrainTrials):
			preTrainRecording[:, i], dW, wNorm = self.preNet.trainStep(stimuli[:, (idx+i)%trainTrials].reshape(-1, 1));

			sumdW += dW;
			sumWnorm += wNorm;
			sumdWSq += dW**2;

			if (i%3000==0 and i!=0):
				print(i, sumdW/(3000), (sumdWSq/3000-(sumdW/3000)**2), sumWnorm/3000);

				sumdW = 0;
				sumdWSq = 0;
				sumWnorm = 0;

				self.preNet.rHH *= 0.8;
				self.preNet.rIH *= 0.8;

			if (np.random.rand()>0.7):
				idx = np.random.randint(preTrainTrials);

		pickle.dump((self.preNet.IH, self.preNet.HH, self.preNet.beta, self.preNet.tau_v), open("trained_net.txt", "wb"))
		self.net = Readout(self.in_size, self.network_size, 4*self.num_ints-1);

		global sumEr, testEr;
		sumEr = [0];
		testEr = [0];

		for i in range(trainTrials):

			trainOut[i], trainRecording[:, i], dWs[i] \
			= self.net.trainStep(stimuli[:, i].reshape(-1, 1), target[:, i].reshape(-1, 1));

			sumEr[-1] += (sums[i] == trainOut[i]);

			if (i%6000==0 and i!=0):

				self.net.rHH *= 0.99;
				self.net.rIH *= 0.99;
				self.net.rHO *= 0.99;

#				self.net.tau_e*= 0.9;
#				self.net.tau_r *= 0.9;

				try:
					self.net.mi *= 0.99;
				except:
					None;

				sumEr[-1] /= 6000;
#				print(i, sumEr[-1]);
				sumEr.append(0);

				for j in range(testTrials):
					testOut[j], testRecording[:, j] \
					= self.net.testStep(stimuli[:, j+trainTrials].reshape(-1, 1));

					testEr[-1] += (sums[j+trainTrials] == testOut[j]);

				testEr[-1] /= testTrials;
				print(i, testEr[-1]);
				testEr.append(0);

			if (i%500==0):
				self.net.update();

		self.net.update();

		fig, (ax1, ax2) = plt.subplots(2);
		ax1.plot(sums[-testTrials:], "go-");


		for i in range(testTrials):
			testOut[i], testRecording[:, i] \
			= self.net.testStep(stimuli[:, i+trainTrials].reshape(-1, 1));

			testEr[-1] += (sums[i+trainTrials] == testOut[i]);

		testEr[-1] /= testTrials;
		print(testEr[-1]);

		ax1.plot(testOut, "bo-");
		ax2.imshow(testRecording);
		ax2.set_aspect('auto');

		return self.net.IH, self.net.HH, sumEr, testEr;

if __name__== "__main__":
	sumErs = [];
	testErs = [];
	for i in range(0,2):
		print(i);
		test = NBack(in_size = 16, num_ints = 3, network_size = 64);
		w_ih, w_hh, sumEr, testEr = test.stimulate(preTrainTrials = i*30000, trainTrials = 600000, testTrials = 4000);
		sumErs.append(sumEr);
		testErs.append(testEr);

	test = NBack(in_size = 16, num_ints = 3, network_size = 64, whole_obj=True);
	w_ih, w_hh, sumEr, testEr = test.stimulate(preTrainTrials = 0, trainTrials = 600000, testTrials = 4000);
	sumErs.append(sumEr);
	testErs.append(testEr);