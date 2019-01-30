# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:,2: 3].values

print(X)
print(y)
# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
#Important here as SVR does not scale features
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)

# Fitting the SVR  Model to the dataset
#kernels can be gaussian, poly, rbf, sigmoid etc
regressor = SVR(kernel='rbf')
regressor.fit(X,y)


# Predicting a new result
# Since the model is fitted on Feature scaled values, to predict we need to scale the feature
#So we fit 6.5 on X fitting parameters which is contained in object sc_X
#StandardScaler().transforms() takes a 2d array as an argument
#We obtain y value(salary) which is in scaled value so we inverse trasform it using sc_y
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Going for a smoother SVR result
X_grid = np.arange(min(X), max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.scatter(X,y,color='red')
plt.title("Truth or Bluff (SVR Precision Model)")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
