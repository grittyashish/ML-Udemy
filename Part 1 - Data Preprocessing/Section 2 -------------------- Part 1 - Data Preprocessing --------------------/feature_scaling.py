import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandarScaler

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Taking care of missing values
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis=0)
imputer = imputer.fit(X[:,1:3])#imputing our concerned columns

X[:,1:3] = imputer.transform(X[:,1:3])

labelEncoder_X = LabelEncoder()
X[:,0] = labelEncoder_X.fit_transform(X[:,0])
#But Label Encoder will lead to placing values like 0,1,2,3, etc which the ML algo may 
#interpret as weights. Thus OneHotEncoder

oneHotEncoder = OneHotEncoder(categorical_features = [0],)
X = oneHotEncoder.fit_transform(X).toarray()
#We can use labelEncoder on y as y is a dependent variable
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)

#Splitting data randomly into training set and testing set
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.5, random_state=42)

print(X_train,y_train,X_test,y_test,sep='\n')

#ML equations use euclidean distance to fit on the data
#Also for ML models like Decision Trees there is no requirement of euclidean distance
#So if we don't scale our model it will run for a very long time.
#Also feature scaling is required to converge the model in a short time.
#Thus we need to scale our features : Feature Scaling
#Types of feature scaling : 
#Standardisation : X_stand = (x-mean(x))/standard_dev(x)
#Normalisation : X_norm = (x-mean(x))/(max(x) - min(x))
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
