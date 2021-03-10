# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 22:50:03 2021

@author: lankuohsing
"""

# In[]
import numpy as np
from computeCostMulti import computeCostMulti
"""
h=\theta_0+\theta_1 * x_1=\theta^T\cdot X
J=1/(2m)*\(X \theta-y)^T(X \theta -y)
\theta_j=\theta_j-\alpha/m*\sumsum_{i=1}{m}(h(x^i)=y^(i))x_j^i
"""
def gradientDescentMulti(X, y, theta, alpha, num_iters):
    """
    X: [m,2]
    y: [m,1]
    theta: [2,1]
    """
    m=y.shape[0]
    J_history=np.zeros((num_iters,1))
    for iter in range(0,num_iters):
        H=np.dot(X,theta)# [m,1]
        Errors=H-y# [m,1]
        theta=theta-alpha/m*np.dot(np.transpose(X),Errors)
        J_history[iter] = computeCostMulti(X, y, theta);
    return theta,J_history