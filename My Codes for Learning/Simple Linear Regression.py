#Univariate Linear Regression

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import math

#Load dataset

# Random data points
X = np.random.randint(5,100, size=50)
y = np.random.randint(5,100, size=50)

# Data points from a dataset
#dataset = pd.read_csv('data.csv')
#X = dataset.iloc[:,0].values
#y = dataset.iloc[:,1].values
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
    ratio = 2/(float(len(X_train)))
    #Iterating over the entire training set
    for x,y in zip(X_train,y_train) : 
        dev_c += (y - (m*x + c) )*(-1)*ratio
        dev_m += (y - (m*x + c) )*(-x)*ratio
    #Now we set the new value of c and m as : 
    c = c - dev_c*alpha
    m = m - dev_m*alpha
    return m,c

def SimpleLinearRegression() : 
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
    cf = CostFunction(m, c, X_train, y_train)
    return m,c,cf
        
if __name__ == "__main__" : 
    iterations = 10000
    print(f"Starting at : m : {m} c : {c} CF : {CostFunction(m,c,X_train,y_train)}")
    for i in range(iterations) : 
        m,c,cf = SimpleLinearRegression()
        print(f"Iteration #{i} m : {m} c : {c} CF : {cf}")

