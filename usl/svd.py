from sklearn.datasets import load_wine
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
# 1) Load dataset
data = load_wine()
X = data.data
y = data.target
print("Original shape:", X.shape)   # (178, 13)
# 2) Apply SVD
svd = TruncatedSVD(n_components=2, random_state=42)
X_svd = svd.fit_transform(X)
print("Reduced shape:", X_svd.shape)
print("Explained variance:", svd.explained_variance_ratio_)
# 3) Visualize SVD result
plt.scatter(X_svd[:,0], X_svd[:,1], c=y, cmap='viridis', s=45)
plt.xlabel("SVD Component 1")
plt.ylabel("SVD Component 2")
plt.title("SVD Projection (Wine Dataset)")
plt.colorbar(label="Class label")
plt.show()
