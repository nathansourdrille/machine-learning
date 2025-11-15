#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 15:51:59 2020

@author: GGilles
"""

## calcul de la solution w
d = 2
w = np.zeros(d)
for i,a in enumerate(alpha):
    w = w + a*Y[i]*X[i,:]
    
## Calcul  de b
# on repere les points support qui sont sur la marge
seuil = 1e-5
# trouver les points tq 0 < alpha < C
idx = np.where((seuil <= alpha )& (alpha <= C - seuil))[0]
n_sv = idx.size
ind_pts_sup_pos = np.where(Y[idx] == 1)[0]
ind_pts_sup_neg = np.where(Y[idx] == -1)[0]
m = min(len(ind_pts_sup_pos), len(ind_pts_sup_neg))

if m==0:
    raise ValueError("Pas de pts supports positifs et negatifs trouve")

# calcul de b  par moyennage
f0pos = X[idx[ind_pts_sup_pos], :].dot(w)
f0neg = X[idx[ind_pts_sup_neg], :].dot(w)
b = -(sum(f0pos[:m]) + sum(f0neg[:m]))/(2*m)