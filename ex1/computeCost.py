# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 23:14:23 2021

@author: lankuohsing
"""
import numpy as np
# In[]
def computeCost(X, y,theta):
    m=y.shape[0]
    J=0
    J=np.sum((y-np.dot(X,theta)**2))/(2*m)
    return J