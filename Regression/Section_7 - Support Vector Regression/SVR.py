# -*- coding: utf-8 -*-
"""
Udemy:
Machine Learnng A-Z

Section 7 - Support Vector Regression
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Feature Scalling
from sklearn.preprocessing import StandardScaler
sc_X  = StandardScaler()
sc_y  = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))


# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(C = 1.0, kernel = 'rbf')
regressor.fit(X, y)

y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

# Visualise the results
plt.scatter(X, y, c = 'red', label = 'True')
plt.plot(X, regressor.predict(X), c = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary ($)')
