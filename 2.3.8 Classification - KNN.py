# K-Nearest Neighbors (k-NN)

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import dataset
dataset = pd.read_csv('2.2 Customer List.csv.csv')
X = dataset.iloc[:, [2,3]].values
Y = dataset.iloc[:, 4].values

# Split dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting k-NN to Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, Y_train)

# Predicting Test set results
Y_pred = classifier.predict(X_test)

var_prob = classifier.predict_proba(X_test)
var_prob[0, :]

# Confusuin Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

64+29