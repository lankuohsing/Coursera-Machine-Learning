# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 22:48:31 2021

@author: lankuohsing
"""

import numpy as np
# In[]
def featureNormalize(X):
    X_norm = X
    mu = np.zeros((1,X.shape[1]))
    sigma = np.zeros((1,X.shape[1]))
    mu=np.mean(X,0)
    sigma=np.std(X,0,ddof=1)
    X_norm=(X-mu)/sigma
    return X_norm, mu, sigma