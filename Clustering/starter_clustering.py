#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 22:10:41 2019

@author: GGilles
"""

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# numpy: matrix computation library
import numpy as np

# sub-package "cluster" of sklearn implements several clustering methods 
# including K-means and Hierachical clustering
from sklearn import cluster

# Useful plotting functions included in the file utility.py (downloadable on Moodle)
from utility import plot_clusters, plot_dendrogram

#%% Synthetic samples : two gaussians data

vmu1 = np.array([0, 0])
vmu2 = np.array([3, 3])
mat_cov = np.eye(2)

n_samples = 20 # number of samples per class
# class 1
X1 = np.random.multivariate_normal(vmu1, mat_cov, n_samples)
Y1 = np.ones(n_samples)
# class 2
X2 = np.random.multivariate_normal(vmu2, mat_cov, n_samples)
Y2 = 2*np.ones(n_samples)

# concatenation of the data
X = np.concatenate((X1, X2))
Y = np.concatenate((Y1, Y2))

# we plot the samples according to their true labels
plot_clusters(X, Y, title = "True data", symbolsize=80)

#%% Applying clustering methods to the samples in X.
# From thereon we will not use the true labels Y. 
# We will only rely on the cluster assignments suggested by the clustering methods

#%% ============== Hierarchical clustering ===================
K = 2 # number of clusters
# defining a hierarchical clustering model with:
# - K=2 clusters, 
# - Euclidean distance as dissimilarity measure
# - "complete (maximal)" linkage between clusters 
hc = cluster.AgglomerativeClustering(n_clusters=K, linkage="complete", metric="l2")

# Fitting and predict the assigned clusters of the samples
labels_hc = hc.fit_predict(X)

# plot the obteained clusters ...
plot_clusters(X, labels_hc, title ='Hierarchical Clustering', symbolsize=80)

# ... and the corresponding dendogram
plot_dendrogram(hc)


#%% K-means 
# we define a K-means model with K = 2 clusters
# the clusters' centers are initialized randomly
# we still keep K=2 clusters
K = 2
kmeans = cluster.KMeans(n_clusters = K, init = "random", random_state=0)

# fit the K-means model on X that is learn the centers of the clusters
kmeans.fit(X)

# predict the assigned clusters
labels_kmeans = kmeans.predict(X)

# plot the obteained clusters and the final centers
plot_clusters(X, labels_kmeans, title ='K-means Clustering', symbolsize=80)
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:, 1], marker='s', s=100);




