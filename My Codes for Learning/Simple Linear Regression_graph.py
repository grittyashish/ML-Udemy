# Univariate Linear Regression

# Giving a graphical intuition of the algorithm

import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import math

# Load dataset
# Random data points
#X = np.random.randint(5,100, size=50)
#y = np.random.randint(5,100, size=50)

# Data points from a dataset
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:,0].values
y = dataset.iloc[:,1].values
X_train = X
y_train = y
m = 0
c = 0
 
def CostFunction(m, c, X_train, y_train) : 
    summation = 0
    for x,y in zip(X_train, y_train) : 
        summation += (y - (m*x + c))**2
    return summation/float(2*len(X_train)) 

#This function will take in alpha,training xs(X_train) and ys(y_train)
#This function will return new m and c for every iteration over the entire training set
#The greater the number of iterations the better the fitting
def GradientDescent(alpha,X_train, y_train) : 
    global m
    global c
    dev_c = 0#Derivative of Cost Function wrt c
    dev_m = 0#Derivative of Cost Function wrt m
    ratio = 1/(float(len(X_train)))
    #Iterating over the entire training set
    for x,y in zip(X_train,y_train) : 
        dev_c += (y - (m*x + c) )*(-1)*ratio
        dev_m += (y - (m*x + c) )*(-x)*ratio
    #Now we set the new value of c and m as : 
    c = c - dev_c*alpha
    m = m - dev_m*alpha
    return m,c

     
############################################################
# Constructing 2-D plot of data points and regression line
###########################################################

fig = plt.figure(figsize=plt.figaspect(.5))
ax_2d = fig.add_subplot(2,1,1)
ax_2d.set(xlim=(0,100), ylim=(0,100))
scatter_plot = ax_2d.scatter(X_train, y_train, c='r' )
x = np.arange(0,100,1)
y = np.zeros((100,))
line = ax_2d.plot(x, y, color='blue', lw=2)[0]

def animate(i) : 
    alpha = 0.0001
    global m
    global c
    #Linear Regression on X_train and y_train
    #Data is simple set of (x,y) points where x is independent variable and y is dependent variable
    #Hypothesis Function is h(x) = mx + c
    #Cost function is : (1/2*len(X_train))*Summation(h(x)-y)^2
    #Summation is over all the training examples
    #Cost funciton is a squared error function
    #We have to minimize the cost function
    #Updation of value : 
    #m = m - alpha*derivativeOfCostFunction
    #c = c - alpha*derivativeofCostFunction
    #The above two steps will be carried uptil there is very slight change in value of m or c
    #or uptil certain steps
    #Above algorithm is called Gradient Descent
    #The number of times this animate function is called is the number of times GD will run and that is the number of steps
    m,c = GradientDescent(alpha, X_train, y_train)
    line.set_ydata([m*x_i + c for x_i in x])
    cf = CostFunction(m, c, X_train, y_train)
    print(f"Frame #{i} m : {m} c : {c} CF : {cf}")
    scatter_point.set_data([m],[c])
    scatter_point.set_3d_properties([cf])
    ax_2d.set_title(f'Iteration #{i} || Cost Function Value : {cf}')
    ax_2d.set_xlabel('X value')
    ax_2d.set_ylabel('Y value')

#########################################
# Constructing 3-D plot of Cost Function
#########################################

ax_3d = fig.add_subplot(2,1,2, projection='3d')
ms = np.linspace(-200,200,100)
bs = np.linspace(-200, 200,100)

M,B = np.meshgrid(ms,bs)
zs = np.array([CostFunction(mp, bp, X_train, y_train) for mp, bp in zip(np.ravel(M), np.ravel(B))])

Z = zs.reshape(M.shape)
ax_3d.plot_surface(M, B, Z, rstride=1, cstride=1, color='b', alpha=0.5)
ax_3d.set_xlabel('m')
ax_3d.set_ylabel('c')
ax_3d.set_zlabel('Cost Function Value')
scatter_point, = ax_3d.plot([m],[c],[CostFunction(m,c,X_train,y_train)],c='r',marker='o')

anim = animation.FuncAnimation(fig, animate, interval = 2000)

plt.draw()
plt.show()

