# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
#Because we want X as a matrix and not a vector or linear array, so [:,1:2]
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#We picked up only one column and that is the column of level as post 
#is unique for all rows and thus doesn't serve any purpose
#Also level is unique for all rows but using level we will produce
#2 further columns by raising the respective levels to some powers

#I will be comparing linear regression Vs polynomial regression here
#For polynomial regression we can vary the degree and by this we see
#That for degree = 4, the model best fits the data

# Fitting Linear Regression to the dataset
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
#Vary the degree for a better understanding
deg = 4
poly_reg = PolynomialFeatures(degree = deg)
X_poly = poly_reg.fit_transform(X)
#Thus X_poly is just a matrix where the values of all the rows of the single column have been taken power of and appended to subsequent columns

#X_poly is a matrix containing values of powers of x : x^0, x^1 and x^2
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#Plotting for Linear Regresssion
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X), color='black')
plt.title('Truth or Bluff Predictor(Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Plotting for Polynomial Regresssion
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg_2.predict(X_poly), color='black')
plt.title('Truth or Bluff Predictor(Polynomial Regression of degree {})'.format(4))
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Plotting for Polynomial Regression 
#Here we will get salaries for fractional levels too
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid)),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='black')
plt.title('Truth or Bluff Predictor(Polynomial Regression of degree {}) and X_grid '.format(4))
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


#Predicting salary for a person who effectively fits in 6.5 level
level = np.array(6.5).reshape(1,1)
print(lin_reg.predict(level))
print(lin_reg_2.predict(level))
#Predicting salary for the same person but using X_grid instead of X for fit_transforming the model
lin_reg_3 = LinearRegression()
lin_reg_3.fit(X_grid,y)
print(lin_reg_3.predict(level))
