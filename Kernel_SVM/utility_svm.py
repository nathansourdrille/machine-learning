#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 12:58:40 2019

@author: GGilles
"""

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

def gen_data_twogaussians_2d(m1, cov1, m2, cov2, n1=25, n2=25):
    
    # generate samples of class 1
    X1 = np.random.multivariate_normal(m1, cov1, n1)
    Y1 = np.ones(n1, dtype=int)
    # class 2
    X2 = np.random.multivariate_normal(m2, cov2, n2)
    Y2 = -1*np.ones(n2, dtype=int)
    # stack samples of both classes
    X = np.concatenate((X1, X2))
    Y = np.concatenate((Y1, Y2))

    return X, Y

def plot_decision_regions_2d(X, Y, classifier, resolution=0.02, title=' '):

    # plot the 2d samples
    plt.figure(figsize=(8,6))
    plt.scatter(X[:,0], X[:, 1], c=Y,  cmap="RdYlBu")
    plt.colorbar(ticks=np.unique(Y))
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 0, X[:, 0].max() + 0
    x2_min, x2_max = X[:, 1].min() - 0, X[:, 1].max() + 0
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                         np.arange(x2_min, x2_max, resolution))
    Z = classifier.decision_function(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    cs = plt.contour(xx1, xx2, Z, levels=np.array([-1, 0, 1]), colors = "g", alpha=0.9, linewidths=4)
    #plt.contour(xx1, xx2, Z)
    plt.clabel(cs, inline=True, fontsize=10)

    plt.title(title, fontsize=14)
    
    
def fetch_data(path, data='train'):
    file_inputs=path+data+'/X_'+data+'.txt'
    df_X = pd.read_csv(file_inputs, header=None, delim_whitespace=True)
    X = df_X.values
    
    file_output=path+data+'/Y_'+data+'.txt'
    df_Y = pd.read_csv(file_output, header=None, delim_whitespace=True)
    Y = df_Y.values[:,0]
    
    return X, Y
