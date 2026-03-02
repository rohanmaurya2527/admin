from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target
print("Original shape:", X.shape)
# Plot histograms
pd.DataFrame(X, columns=data.feature_names).hist(
    bins=20, figsize=(18,15), color="steelblue", edgecolor="black"
)
plt.suptitle("Histograms of All Features (Breast Cancer)")
plt.tight_layout()
plt.show()
# Standardize + PCA (2 components)
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print("Reduced shape:", X_pca.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance captured:", pca.explained_variance_ratio_.sum())
# 2D PCA Plot
plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='coolwarm', alpha=0.8)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection (2D) — Breast Cancer Dataset")
plt.legend(*scatter.legend_elements(), title="Classes")
plt.grid(alpha=0.3)
plt.show()
