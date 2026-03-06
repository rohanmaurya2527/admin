import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Load dataset
data = load_iris()
X = data.data

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------- Elbow Method --------
inertia = []
for k in range(1,9):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertia.append(model.inertia_)

plt.plot(range(1,9), inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.grid()
plt.show()

# -------- Silhouette Score --------
scores = []
for k in range(2,7):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    scores.append(score)

plt.plot(range(2,7), scores, marker='o')
plt.title("Silhouette Score vs k")
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.grid()
plt.show()

# -------- Final KMeans Model --------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
labels = kmeans.fit_predict(X_scaled)

# MiniBatchKMeans
mbk = MiniBatchKMeans(n_clusters=3, random_state=42, batch_size=20)
mbk_labels = mbk.fit_predict(X_scaled)

# -------- PCA Visualization --------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='tab10')
centers = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers[:,0], centers[:,1], c='black', marker='x', s=120)

plt.title("KMeans Clusters (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid()
plt.show()

# -------- Evaluation --------
print("KMeans Inertia:", kmeans.inertia_)
print("Silhouette Score (KMeans):", silhouette_score(X_scaled, labels))
print("Silhouette Score (MiniBatch):", silhouette_score(X_scaled, mbk_labels))

# -------- Predict New Sample --------
new_sample = np.array([[5.0, 3.2, 1.2, 0.2]])
new_sample_scaled = scaler.transform(new_sample)

cluster = kmeans.predict(new_sample_scaled)[0]
print("New sample assigned to cluster:", cluster)
