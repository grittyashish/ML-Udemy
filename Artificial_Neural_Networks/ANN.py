# Artificial Neural Network

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data

#Encoding City
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

#Encoding Gender
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#For City as there are 3 values for this categorical variable
#Thus we have to use one hot encoder
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

#Excluding one Dummy Variable to avoid Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Initialising the ANN
# ANN is a classifier here
classifier = Sequential()

# We have 11 independent variables in this problem
# ReLu is best for input layer and hidden layers
# Sigmoid is best for output layer as it gives us probabilities for each of the observation of the test set.

#There is no rule of thumb in choosing the number of nodes in the hidden layer
#The best practice is : #hidden_layer_nodes = avg(#input_layer_nodes,#output_layer_nodes)
#Otherwise use parameter tuning to get the #hidden_layer_nodes

# Adding the input layer and the first hidden layer
# By the above formula : 11(input layer nodes) + 1(output layer node) = 12/2 = 6(hidden layer node)
# Thus output_dim = 6
# output_dim also specifies(by default) the number of units in the layer
# init = 'uniform' : initialize the weights according to uniform distribution

# The first use of .add() adds the first hidden layer.
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
# This layer will automatically know what the number of units are there in previous layer so no input_dim
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
# Using sigmoid activation function
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
# There is no rule of thumb for epoch and batch_size
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# If probability > 0.5 then positive class : Customer will leave
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
