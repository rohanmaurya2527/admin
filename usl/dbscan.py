from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# Load and scale data
iris=load_iris()
X=iris.data
sc=StandardScaler()
X = sc.fit_transform(X)
# Apply DBSCAN
dbscan = DBSCAN(eps=0.8, min_samples=5)
labels = dbscan.fit_predict(X)
# Reduce to 2D using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
# Plot results
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("DBSCAN Clustering")
plt.show()
