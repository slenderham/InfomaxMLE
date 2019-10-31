# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
import itertools

def STA(spike, input_seq, n_back):
	spike_freq = np.zeros((spike.shape[0], 10**n_back));

#	digs = [str(i) for i in range(10)];
#	ticks = [''];
#	for i in range(n_back):
#		ticks = list(itertools.product(ticks, digs));


	for i in range(n_back, input_seq.shape[0]):
		index = 0;
		for j in range(n_back):
			index += input_seq[i-j]*10**j;
		spike_freq[:,int(index)] += spike[:,i];

	spike_freq /= np.sum(spike_freq, axis=1, keepdims=True);
	plt.imshow(spike_freq);
	fig, axes = plt.subplots(8,8);
	maxNum = np.max(np.max(spike_freq));


	for i in range(64):
		axes[i//8, i%8].bar(range(10**n_back), spike_freq[i,:]);
		axes[i//8, i%8].set_ylim(0, maxNum*1.05);
		axes[i//8, i%8].tick_params(axis='both',labelsize=8)


if __name__=="__main__":
	STA(testRecording, testStimuli, 2);