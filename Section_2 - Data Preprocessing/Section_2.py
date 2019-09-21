# -*- coding: utf-8 -*-
"""
Udemy:
Machine Learning A-Z

Section 2 - Data Preprocessing

"""

import os
os.chdir('C:\\Users\\hz336yw\\Downloads\\Machine_Learning_AZ\\Machine-Learning-A-Z-New\\Machine Learning A-Z New\\Part 1 - Data Preprocessing\\Section 2 -------------------- Part 1 - Data Preprocessing --------------------')

import pandas as pd

# Importing the dataset
data = pd.read_csv("Data.csv")

# Split to features and dependent variable
X = data.iloc[:,0:3].values
y = data.iloc[:,3].values

# Handling missing data through imputation
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.fit_transform(X[:,1:3])

# Encoding the categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder = LabelEncoder()
X[:,0] = labelencoder.fit_transform(X[:,0])

onehotenc_x = OneHotEncoder(categorical_features = [0])
X = onehotenc_x.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
