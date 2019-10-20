"""
Udemy:
Machine Learnng A-Z

Section 6 - Polynomial Regression

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting a linear regression
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
linreg.fit(X, y)

# Fitting a polynomial model
from sklearn.preprocessing import PolynomialFeatures

polyreg = PolynomialFeatures(degree = 4)
X_polynomial = polyreg.fit_transform(X)
# The first column that its being creates corresponds to the constant (b0)

linreg_2 = LinearRegression()
linreg_2.fit(X_polynomial, y)

# Visualising the Linear Regression results

plt.scatter(X, y, c = 'red', label = 'True') # Real values
plt.plot(X, linreg.predict(X), c = 'blue', label = 'Predicted (Linear Regression)' ) # Predicted salaries from the linreg model
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary ($)')
plt.legend(loc = 'best')

# Visualising the Polynomial Regression results

plt.scatter(X, y, c = 'red', label = 'True')
plt.plot(X, linreg_2.predict(X_polynomial), c = 'green', label = 'Predicted (Polynomial)')
plt.title('Truth or Bluff (4th Degree Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary ($)')
plt.legend(loc = 'best')

# Predict a new result
linreg.predict(np.array(6.5).reshape(1,-1))
