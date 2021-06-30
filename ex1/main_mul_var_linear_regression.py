# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 22:48:04 2021

@author: lankuohsing
"""
"""
多变量线性回归
"""
# In[]
import numpy as np
from featureNormalize import featureNormalize
from computeCostMulti import computeCostMulti
from gradientDescentMulti import gradientDescentMulti
from normalEqn import normalEqn
import matplotlib.pyplot as plt
# In[]
X=[]
Y=[]
with open("./ex1data2.txt",'r',encoding="UTF-8") as rf:
    for line in rf:
        splitResults=line.strip().split(",")
        X.append([float(splitResults[0]),float(splitResults[1])])
        Y.append(float(splitResults[2]))
# In[]
X=np.array(X)
Y=np.array(Y).reshape(len(Y),1)
m=X.shape[0]
print("X=[",X[0:10,:],"],Y=[",Y[0:10,:],"]")
# In[]
X_norm, mu, sigma = featureNormalize(X);

# In[]

X_norm=np.concatenate((np.ones((m,1)),X_norm),axis=1)
theta=np.zeros((3,1))
num_iters=400
alpha=0.1

# In[]
J=computeCostMulti(X_norm, Y, theta)
theta,J_history = gradientDescentMulti(X_norm, Y, theta, alpha, num_iters)

# In[]
X=np.array(X)
Y=np.array(Y).reshape(len(Y),1)
m=X.shape[0]
X=np.concatenate((np.ones((m,1)),X),axis=1)
print("X=[",X[0:10,:],"]\n,Y=[",Y[0:10,:],"]")
theta=normalEqn(X,Y)
print("Theta computed from normal equation:",theta[0],theta[1],theta[2])