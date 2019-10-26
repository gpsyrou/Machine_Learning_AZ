# -*- coding: utf-8 -*-
"""
K - Means Clustering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the dataset
# Spending score 1-->100 (low to high)
# We want to cluster the customers
data = pd.read_csv('Mall_Customers.csv')

X = data.iloc[:, [3,4]].values

# K-means algorithms
from sklearn.cluster import KMeans

# Find the optinal number of clusters through WCSS
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10
                    , verbose = 0, random_state = 0)
     
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Apply K-means algorithm
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10
                    , verbose = 0, random_state = 0)

# Identify to which cluster each customer belongs to
y_kmeans = kmeans.fit_predict(X)

# Visualising the Clustering results
plt.figure(figsize=(8,10))

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 50, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 50, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 50, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 50, c = 'lightgreen', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 50, c = 'orange', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s  = 100, c = 'magenta', label = 'Centroids')
plt.title('Clusters of clients - K-means Algorithm')
plt.xlabel('Annual Income ($)')
plt.ylabel('Spending Score (1-100)')
plt.legend(loc='best')
plt.grid(True, alpha = 0.3)
plt.show()
