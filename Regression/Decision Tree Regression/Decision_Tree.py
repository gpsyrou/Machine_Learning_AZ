"""
Udemy:
Machine Learnng A-Z

Section 8 - Decision Trees Regression
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Fit the Decision Tree regressor
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

y_pred = regressor.predict(np.array([[6.5]]))

# Visualise the results
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red', label = 'True')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue', label = 'Predicted')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary ($)')
plt.legend(loc = 'best')
plt.show()
