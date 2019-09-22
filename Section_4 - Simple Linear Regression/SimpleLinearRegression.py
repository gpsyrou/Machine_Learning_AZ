# -*- coding: utf-8 -*-
"""
Udemy:
Machine Learning A-Z

Section 4 - Simple Linear Regression
"""

import os
os.chdir("C:\\Users\\hz336yw\\Downloads\\Machine_Learning_AZ\\Machine-Learning-A-Z-New\\Machine Learning A-Z New\\Part 2 - Regression\\Section 4 - Simple Linear Regression")

# Import data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("Salary_Data.csv")

# Split the data to independent(years of experience) and dependend variables(salary)
X = data.iloc[:, :-1].values 
y = data.iloc[:, 1].values

# Split to train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fitting a Linear Regression model to the training data
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

# Predicting values for the test set
y_hat = linear_regressor.predict(X_test)

# Visualising the results
# Training Set
plt.figure(figsize=(10,6))

plt.scatter(X_train, y_train, color = 'red', label = 'Real values (yi)')
plt.plot(X_train, linear_regressor.predict(X_train), color = 'blue', label = 'Estimated Regression Line, f(x) = b0 +b1*x') # estimated regression line
plt.title("Salary($) vs Years of Experience (Training set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.grid(True)
plt.legend(loc = 'best')
plt.show()

# Test Set
plt.figure(figsize=(10,6))

plt.scatter(X_test, y_test, color = 'red', label = 'Real values (yi)')
plt.plot(X_train, linear_regressor.predict(X_train), color = 'blue', label = 'Estimated Regression Line, f(x) = b0 +b1*x') # estimated regression line
plt.title("Salary($) vs Years of Experience (Test set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.grid(True)
plt.legend(loc = 'best')
plt.show()  

# Evaluate performance
# Methods: MAE, MSE, RMSE
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test, y_hat)
mse = mean_squared_error(y_test, y_hat)

print('Mean Absolute Error: {0}'.format(mae))  
print('Mean Squared Error: {0}'.format(mse)) 
print('Root Mean Squared Error: {0}'.format(np.sqrt(mse)))

from sklearn.metrics import r2_score
print('R-squared: {0}'.format(r2_score(y_test, y_hat)))
