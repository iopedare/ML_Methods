# Polynomial Regression

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import dataset
dataset = pd.read_csv('2.2 Gaming_data.csv.csv')
X = dataset.iloc[:, 0:1].values
Y = dataset.iloc[:, 1].values

# Fitting Linear Regression to dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

# Fitting Polynomial Regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, Y)

# Visualising Linear Regression results
plt.scatter(X,Y)
plt.plot(X, lin_reg.predict(X), color='red')
plt.title('Gaming Data (Linear Regression)')
plt.xlabel('Gaming Steps')
plt.ylabel('Points')
plt.show()

# VIsualizing Polynomial Regression results
plt.scatter(X,Y)
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color='red')
plt.title('Gaming Data (Polynomial Regression)')
plt.xlabel('Gaming Steps')
plt.ylabel('Points')
plt.show()

# Predicting new result with Linear Regression
lin_reg2.predict(poly_reg.fit_transform([[11]]))








