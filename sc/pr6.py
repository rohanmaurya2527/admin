#Code 1:
import numpy as np
class SOM:
    def __init__(self, grid_size, input_dim, learning_rate=0.5, radius=2):
        self.grid_size = grid_size
        self.input_dim = input_dim
        self.lr = learning_rate
        self.radius = radius
        # Initialize weights randomly (grid_size x grid_size x input_dim)
        self.weights = np.random.rand(grid_size, grid_size, input_dim)
        # Precompute neuron grid coordinates
        self.grid_x, self.grid_y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    def find_bmu(self, x):
        """Find Best Matching Unit (BMU) index for input x"""
        diff = self.weights - x  # Broadcast subtraction
        dist = np.linalg.norm(diff, axis=2)
        return np.unravel_index(np.argmin(dist), dist.shape)
    def train(self, data, epochs=10):
        for epoch in range(epochs):
            # Optional: decay learning rate and radius over time
            lr = self.lr * np.exp(-epoch / epochs)
            radius = self.radius * np.exp(-epoch / epochs)
            for x in data:
                bmu = self.find_bmu(x)
                # Compute distance from BMU for all neurons
                dist_to_bmu = np.sqrt((self.grid_x - bmu[0])**2 + (self.grid_y - bmu[1])**2)
                # Compute neighborhood influence (Gaussian)
                influence = np.exp(-(dist_to_bmu**2) / (2 * (radius**2)))
                # Update all weights at once (broadcasting)
                self.weights += lr * influence[:, :, np.newaxis] * (x - self.weights)
# ------------------ Usage ------------------
# Random dataset (100 samples, 3 features)
data = np.random.rand(100, 3)
# Create SOM (5x5 grid, 3D input)
som = SOM(grid_size=5, input_dim=3, learning_rate=0.5, radius=2)
# Train SOM for 10 epochs
som.train(data, epochs=10)
print("Training complete!")
print("Final SOM weights:\n", som.weights)

#Code 2: with Iris dataset
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
class SOM:
    def __init__(self, grid_size, input_dim, learning_rate=0.5, radius=2):
        self.grid_size = grid_size
        self.input_dim = input_dim
        self.lr = learning_rate
        self.radius = radius
        self.weights = np.random.rand(grid_size, grid_size, input_dim)
    def find_bmu(self, x):
        diff = self.weights - x
        dist = np.linalg.norm(diff, axis=2)
        return np.unravel_index(np.argmin(dist), dist.shape)
    def train(self, data, epochs=200):
        for epoch in range(epochs):
            for x in data:
                bmu = self.find_bmu(x)
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        dist = np.sqrt((i - bmu[0])**2 + (j - bmu[1])**2)
                        if dist <= self.radius:
                            influence = np.exp(-(dist**2) / (2 * (self.radius**2)))
                            self.weights[i, j] += influence * self.lr * (x - self.weights[i, j])
# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
# Normalize
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)
# Train SOM
som = SOM(grid_size=20, input_dim=4, learning_rate=0.5, radius=2)
som.train(X_norm, epochs=300)
print("Training completed!")
# Mapping samples to SOM grid
mapping = np.zeros((20, 20), dtype=int)
for i, x in enumerate(X_norm):
    bmu = som.find_bmu(x)
    mapping[bmu] = y[i] + 1
print("\nSOM Mapping of Iris Classes:")
print("(1=setosa, 2=versicolor, 3=virginica)")
print(mapping)
