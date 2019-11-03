"""
Natural Language Processing
Bag of Words model
"""

# Creating a Bag of Words model for Restaurant Reviews

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the data
data = pd.read_csv('Restaurant_Reviews.tsv', sep = '\t', header = [0])

# Cleaning the text
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

stopWords = set(stopwords.words('english'))

# Remove pancuation in each line
data['Review'] = data['Review'].apply(lambda text: re.sub(r'[^\w\s]','',text))

# Make all strings lowercase and split into list of words per row
data['Review'] = data['Review'].apply(lambda text: text.lower().split())

# Remove stopwords from each line and perform Stemming
ps = PorterStemmer()
data['Review'] = data['Review'].apply(lambda text: 
    [ps.stem(word) for word in text if word not in stopWords])

# Finally make the words to join back and form a 'string' per row
data['Review'] = data['Review'].apply(lambda ls: ' '.join(ls))

# Our final Corpus
corpus = data['Review']

# Creating the BagOfWords model

# This model will create one column per word
# Columns: Words
# Rows: Reviews 

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) # Keep the 1500 most relevant words
X = cv.fit_transform(corpus).toarray()

X.shape # 1000 reviews (rows) and 1500 words (columns)

y = data.iloc[:, 1].values


# Training a Naive Bayes Classification model to our corpus

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes 
from sklearn.naive_bayes import GaussianNB
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)

# Make predictions on the test set
y_pred = naive_bayes.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

'''
TP:56
FP:41
FN:13
TN:90
'''

'''
Accuracy = (TP + TN) / (TP + TN + FP + FN)

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1 Score = 2 * Precision * Recall / (Precision + Recall)
'''

def GetEvaluationMetrics(confusion_matrix, metric = 'all'):
    '''
    Takes as input the results of a confusion matrix, and calculates
    the Accuracy, Precision, Recall and F1 Score of a classification
    algorithm
    
    Parameters: 
        confusion_matrix: sklearn.metrics.classification.confusion_matrix object
        metric: 'all','accuracy','precision','recall',f1_score'. Default is 'all'
    '''
    size = np.sum(confusion_matrix)
    accuracy = confusion_matrix[0][0] + confusion_matrix[1][1] / size
    precision = np.round(100 * (confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])), 2)
    recall = np.round(100 * confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0]), 2)
    f1_score = np.round(((2 * precision * recall) / (precision + recall)) / 100 , 3)
    
    print(f'The accuracy is {accuracy}%\nThe precision is {precision}%\nThe recall is {recall}%\nThe F1-score is {f1_score}')
    
    return None

GetEvaluationMetrics(cm)

