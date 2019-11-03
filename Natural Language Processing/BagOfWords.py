"""
Natural Language Processing
"""

# Creating a Bag of Words model for Restaurant Reviews

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the data
data = pd.read_csv('Restaurant_Reviews.tsv', sep = '\t', header = [0])

# Cleaning the text
from nltk.corpus import stopwords
import re

stopWords = set(stopwords.words('english'))

# Remove pancuation in each line
data['Review'] = data['Review'].apply(lambda text: re.sub(r'[^\w\s]','',text))

# Make all strings lowercase and split into list of words per row
data['Review'] = data['Review'].apply(lambda text: text.lower().split())

# Remove stopwords from each line
data['Review'] = data['Review'].apply(lambda text: [x for x in text if x not in stopWords])















