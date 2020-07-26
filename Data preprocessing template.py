# Data Pre-processing

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import dataset
dataset = pd.read_csv('2.2 Data.csv.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# Split dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.20,random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
