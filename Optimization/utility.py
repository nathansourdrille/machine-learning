#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 17:36:18 2019

@author: GGilles
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_contours_exercice_section2(A, b, c):
    if c.shape[0] !=2:
        raise "The contour plot is only available for 2-dimensional QP problems"
        
    #  contour plot of the objective function
    n = 100
    points_x1, points_x2 = np.meshgrid(np.linspace(-2, 5, n), np.linspace(-2, 5, n))
    J = 0.5*((points_x1 - c[0])**2) + 0.5*((points_x2 - c[1])**2)
    J = J.reshape(points_x1.shape)
    cp = plt.contour(points_x1, points_x2, J, 15, alpha=0.95, linewidths=2, cmap=cm.Set1)
    plt.clabel(cp, inline=True, fmt='%1.1f', fontsize=10)
    plt.plot(c[0], c[1], 's', color='b', markerfacecolor='b', markersize = 10)
    
    # contour plot of the constraints
    points_x1, points_x2 = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    pt_x1 = points_x1.reshape(n*n)
    pt_x2 = points_x2.reshape(n*n)
    ineq = A.dot(np.vstack((pt_x1, pt_x2))) - np.outer(b, np.ones(n*n))
    cp = plt.contour(points_x1, points_x2, ineq[0,:].reshape(n,n), [0], alpha=1, linewidths=2, colors='k')
    cp = plt.contour(points_x1, points_x2, ineq[1,:].reshape(n,n), [0], alpha=1, linewidths=2, colors='k')
    cp = plt.contour(points_x1, points_x2, ineq[2,:].reshape(n,n), [0], alpha=1, linewidths=2, colors='k')
    cp = plt.contour(points_x1, points_x2, ineq[3,:].reshape(n,n), [0], alpha=1, linewidths=2, colors='k')
