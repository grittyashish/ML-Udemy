# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#Add a column of ones in the matrix feature X
#Why? To provide a bias value just like y-intercept kind of thing for
#Multiple regression.
#The same was done in case of linear regression but it was being done by default
#By the scikit learn library
X = np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)

#Now using backward elimination to eliminate the variables which have less statistical inference
import statsmodels.formula.api as sm

def backward_elimination(X,sl) : 
    #Get number of independent variables
    numVars = len(X[0])
    
    for i in range(numVars) : 
        regressor_OLS = sm.OLS(endog=y,exog=X).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl : 
            for j in range(numVars - i) : 
                if regressor_OLS.pvalues[j].astype(float) == maxVar : 
                    X = np.delete(X, j, 1)
    
    print(regressor_OLS.summary())
    return X

#X_opt is matrix of optimal features
X_opt = X[:,[0,1,2,3,4,5]]
SL = 0.05
X_backwardEliminationModelled = backward_elimination(X_opt, SL)

