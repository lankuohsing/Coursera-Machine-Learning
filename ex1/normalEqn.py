# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 23:04:35 2021

@author: lankuohsing
"""

import numpy as np
# In[]
def normalEqn(X,y):
    m=X.shape[0]
    theta=np.zeros((m,1))
    theta = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,y))
    return theta
