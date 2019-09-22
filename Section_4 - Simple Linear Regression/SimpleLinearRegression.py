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


