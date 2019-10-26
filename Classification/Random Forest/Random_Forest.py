"""
Random Forest Classification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the data
data = pd.read_csv('Social_Network_Ads.csv')

# ID is not a useful feature
data.drop(columns = ['User ID'], inplace = True)

# Split the data to dependent and independent variables
X = data.iloc[:, 0:3].values
y = data.iloc[:, 3].values

# Encode the categorical variable for Sex ('Female', 'Male')
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_enc = LabelEncoder()
X[:, 0] = label_enc.fit_transform(X[:,0])

onehotenc_x = OneHotEncoder(categorical_features = [0])
X = onehotenc_x.fit_transform(X).toarray()


# Split to train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.25, random_state = 0)


# Fit the decision trees and random forest algorithms
from sklearn.ensemble import RandomForestClassifier
rf_class = RandomForestClassifier(n_estimators = 20)

rf_class.fit(X_train, y_train)

estimator = rf_class.estimators_[3]

# Make predictions
y_pred = rf_class.predict(X_test)

# Visualise results
from sklearn.metrics import confusion_matrix
confusion_matrix(y_true = y_test, y_pred = y_pred)

from sklearn.tree import export_graphviz

export_graphviz(estimator, out_file='tree.dot', 
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')

