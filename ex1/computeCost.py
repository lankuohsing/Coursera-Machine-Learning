# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 23:14:23 2021

@author: lankuohsing
"""
# In[]
import numpy as np

# In[]
def computeCost(X,Y,theta):
    m=Y.shape[0]
    J=0
#    J=np.sum((Y-np.dot(X,theta))**2)/(2*m)
    J=np.dot(np.transpose(Y-np.dot(X,theta)),Y-np.dot(X,theta))/(2*m)
    return J