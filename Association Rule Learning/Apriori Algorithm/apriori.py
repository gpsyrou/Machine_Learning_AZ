# -*- coding: utf-8 -*-
"""
Association Rules - Apriori Algorithm
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from apyori import apriori

# Import the data
data = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

# Each row is a customer transaction, and the products that they bought
# The Apriori algorithm expects a list of lists, where each list is a row
# that contains the products as strings
transactions = []
for i in range(0,len(data)):
    transactions.append([str(data.values[i,j]) for j in range(0,len(data.columns))])

# Train the algorithm
rules = apriori(transactions, min_support = 0.003 ,
                min_confidence = 0.2, min_lift = 3, min_length = 2)
    
# Visualising the results
results = list(rules) 
