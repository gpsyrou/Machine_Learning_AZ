"""
Hierarchical Clustering (Agglomerative)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import Data
data = pd.read_csv('Mall_Customers.csv')
X = data.iloc[:, [3,4]].values

# Using Dendrograms to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward')) # ward: minimize the variance within each cluster
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

# Optimal number of clusters is 5

# Fit a hierarchical clustering algorithm
from sklearn.cluster import AgglomerativeClustering
agl_clust = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = agl_clust.fit_predict(X)

# Visualising the results

plt.figure(figsize=(8,10))

plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 50, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 50, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 50, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 50, c = 'lightgreen', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 50, c = 'orange', label = 'Cluster 5')
plt.title('Clusters of clients - Agglomerative Hierarchical Clustering')
plt.xlabel('Annual Income ($)')
plt.ylabel('Spending Score (1-100)')
plt.legend(loc='best')
plt.grid(True, alpha = 0.3)
plt.show()
