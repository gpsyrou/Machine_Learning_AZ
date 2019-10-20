# -*- coding: utf-8 -*-
"""
Udemy:
Machine Learnng A-Z

Section 9 - Random Forest Regression
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fit the model
from sklearn.ensemble import RandomForestRegressor
ranfor_reg = RandomForestRegressor(n_estimators = 300, criterion = 'mse'
                                   ,random_state = 0)
ranfor_reg.fit(X, y)


y_pred = ranfor_reg.predict(np.array([[6.5]]))

# Visualise the results
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red', label = 'True')
plt.plot(X_grid, ranfor_reg.predict(X_grid), color = 'blue', label = 'Predicted')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary ($)')
plt.legend(loc = 'best')
plt.show()
