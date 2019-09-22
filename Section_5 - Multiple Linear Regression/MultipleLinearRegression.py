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

y = b0 + b1*x1 + b2*x2 + ... + bn * xn

Assumptions of Linear Regression:
    1) Linearity
    2) Homoscedasticity
    3) Multivariate Normality
    4) Independence of errors
    5) Lack of multicollinearity
'''

# Split dataset to dependent and independent variables
X = data.iloc[:, 0:4]
y = data.iloc[:, -1]


# Creating dummy variables for State
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

