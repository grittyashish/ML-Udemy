import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
print(X)
y = dataset.iloc[:,-1].values
print(y)

#Taking care of missing values
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis=0)
imputer = imputer.fit(X[:,1:3])#imputing our concerned columns

X[:,1:3] = imputer.transform(X[:,1:3])

labelEncoder_X = LabelEncoder()
X[:,0] = labelEncoder_X.fit_transform(X[:,0])
print(X)
#But Label Encoder will lead to placing values like 0,1,2,3, etc which the ML algo may 
#interpret as weights. Thus OneHotEncoder

oneHotEncoder = OneHotEncoder(categorical_features = [0],)
X = oneHotEncoder.fit_transform(X).toarray()
print(X)
#We can use labelEncoder on y as y is a dependent variable
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)


