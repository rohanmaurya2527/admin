from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np

# Load and scale data
iris=load_iris()
X=iris.data
sc=StandardScaler()
X = sc.fit_transform(X)

# Create dendrogram
linked = linkage(X, method='ward')
dendrogram(linked, truncate_mode='level', p=5)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

# Apply Agglomerative Clustering
agc = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels=agc.fit_predict(X)
print("Cluster Labels:", np.unique(labels))

# PCA visualization
# Reduce to 2D using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Agglomerative Clustering")
plt.show()
