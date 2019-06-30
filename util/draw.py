#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 22:51:10 2019

@author: wangchong
"""
import numpy as np
from matplotlib import pyplot as plt

def draw_fig(ax1, ax2, ax3, ss, gg, trainTarget, trainOut, trainRecording, w_dot):
    trainTime = np.arange(ss, gg);    
#    plt.plot(trainTime[ss:gg], u[ss:gg], color=plt.cm.Blues(0.3), linewidth=4, label='$input$')
    ax1.plot(trainTime, trainTarget[ss:gg], color=plt.cm.Greens(0.6), linewidth=2)
    ax1.plot(trainTime, trainOut[ss:gg], 'r')
    ax1.set_title('Learning')
    ax1.legend(loc=3, fancybox=True, framealpha=0.7, fontsize='large')
    ax1.tick_params(labelsize=12)

    ax2.plot(trainTime, trainRecording[2, ss:gg], color=plt.cm.Oranges(0.5))
    ax2.plot(trainTime, trainRecording[10, ss:gg] , 'b')
    ax2.plot(trainTime, trainRecording[70, ss:gg] , color=plt.cm.Purples(0.6))
    ax2.plot(trainTime, trainRecording[100, ss:gg] , 'm')

    ax2.set_xlabel('Time (msec)').set_fontsize(14)
    ax2.legend(loc=3, fancybox=True, framealpha=0.7, fontsize='large')
    ax2.tick_params(labelsize=12)
    # #
    ax3.plot(trainTime, w_dot[ss:gg], 'b');
    
    plt.draw();
    plt.pause(0.1);