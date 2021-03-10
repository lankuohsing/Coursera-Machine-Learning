# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 22:48:56 2021

@author: lankuohsing
"""

# In[]
import numpy as np

# In[]
def computeCostMulti(X,Y,theta):
    m=Y.shape[0]
    J=0
    J=np.dot(np.transpose(Y-np.dot(X,theta)),Y-np.dot(X,theta))/(2*m)
    return J