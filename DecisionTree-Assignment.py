# Decision Tree

# Import libraries
import numpy as np
import pandas as pd


# Import dataset
dataset = pd.read_csv('Iris.csv')


#mapping the dependent variable
dataset['Species'] = dataset['Species'].map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})

#Selecting the independent and dependent variables
X = dataset.iloc[:, [1,2,3,4]].values
Y = dataset.iloc[:, [5]].values

# Split dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)


# Fitting Decision Tree to Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, Y_train)


# Predicting Test set results
Y_pred = classifier.predict(X_test)

# Confusuin Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

#Accuracy
accuracy = ((16+17+11)/45)*100
accuracy