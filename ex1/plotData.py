# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 23:15:46 2021

@author: lankuohsing
"""

# In[]
import matplotlib.pyplot as plt
import numpy as np
# In[]
def plotData(X,Y,xLabel="", yLabel="", title=""):
    plt.plot(X,Y,'r.')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    plt.show()


if __name__=="__main__":
    X = np.arange(-20., 20., 0.1)
    Y=100*X+2*X**2-1*X**3+5
    plotData(X,Y,"X","X+2X^2+3X^3+4",title="polynomial")