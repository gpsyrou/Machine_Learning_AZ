# -*- coding: utf-8 -*-
"""
Udemy:
Machine Learnng A-Z

Section 5 - Multiple Linear Regression

"""

import os
os.chdir("C:\\Users\\george\\Desktop\\temp\\Multiple_Linear_Regression")

# Import dataset
import pandas as pd

data = pd.read_csv("50_Startups.csv")
data.columns
data.dtypes

# 'Profit' is the dependent variable. Rest of them are the features.
# 'State' is a categorical variable. Thus we need to use a dummy variable for our LR model.

'''

y = b0 + b1*x1 + b2*x2 + ... + bn * xn + bN+1 * Dummy1

Assumptions of Linear Regression:
    1) Linearity
    2) Homoscedasticity
    3) Multivariate Normality
    4) Independence of errors
    5) Lack of multicollinearity
'''

# Split dataset to dependent and independent variables
X = data.iloc[:, 0:4].values
y = data.iloc[:, -1].values


# Creating dummy variables for State
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder = LabelEncoder()
X[:,3] = label_encoder.fit_transform(X[:,3])

onehotencoder  = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()


# Avoiding the Dummy Variable Trap (3 states so we take n-1 dummies)
X = X[:, 1:]

 
# Split into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fit multiple Linear Regression models to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions
y_hat = regressor.predict(X_test)




