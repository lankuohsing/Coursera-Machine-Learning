# In[]
import numpy as np
from plotData import plotData
from computeCost import computeCost
from gradientDescent import gradientDescent
# In[]
X=[]
Y=[]
with open("./ex1data1.txt",'r',encoding="UTF-8") as rf:
    for line in rf:
        splitResults=line.strip().split(",")
        X.append(float(splitResults[0]))
        Y.append(float(splitResults[1]))
# In[]
X=np.array(X)
Y=np.array(Y)

# In[]
plotData(X,Y,xLabel="X",yLabel="Y",title="Linear Regression")
# In[]
m=X.shape[0]
X=np.concatenate((np.ones((m,1)),X.reshape(m,1)),axis=1)
Y=Y.reshape((m,1))
theta=np.zeros((2,1))
iterations=1500
alpha=0.01

# In[]
J=computeCost(X, Y, theta)
theta,J_history = gradientDescent(X, Y, theta, alpha, iterations)

# In[]
print("Theta computed from gradient descent:",theta[0],theta[1])