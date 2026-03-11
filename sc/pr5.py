#Code 1: With hardcoded example
import numpy as np
def hebbian_learning(X, y, lr=0.1):
    """
    X : Input samples (num_samples × num_features)
    y : Output values for each sample
    lr: Learning rate
    """
    # Step 1: Initialize weights to zero
    weights = np.zeros(X.shape[1])
    # Step 2–5: Loop through each training sample
    for i in range(len(X)):
        x_i = X[i]
        y_i = y[i]
        # Step 3: Calculate weight change
        delta_w = lr * x_i * y_i
        # Step 4: Update weights
        weights += delta_w
        print(f"After sample {i+1}, updated weights: {weights}")
    return weights
# Example usage
X = np.array([[1, 0],[0, 1],[1, 1],[0, 0]])
y = np.array([0, 0, 1, 0])  # desired outputs
final_weights = hebbian_learning(X, y)
print("\nFinal learned weights:", final_weights)

#Code 2: With Iris dataset
from sklearn.datasets import load_iris
import numpy as np
# Step 1: Load Iris data
iris = load_iris()
X = iris.data
y = iris.target   # Classes: 0, 1, 2
# Step 2: Convert to +1 / -1 target for Hebbian rule
# Here: class 1 becomes -1, all others become +1
y_heb = np.where(y == 1, -1, 1)
# Step 3: Normalize input (min-max normalization)
X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# Step 4: Initialize weights
weights = np.zeros(X_norm.shape[1])
learning_rate = 0.1
print("Initial weights:", weights)
# Step 5: Hebbian Learning
for i in range(len(X_norm)):
    x_i = X_norm[i]
    y_i = y_heb[i]
    # Hebbian update
    delta_w = learning_rate * x_i * y_i
    weights += delta_w
    # Show step-by-step for first 5 samples
    if i < 5:
        print(f"\nSample {i+1}")
        print("Input:", x_i)
        print("Output (Hebbian label):", y_i)
        print("Delta w:", delta_w)
        print("Updated weights:", weights)
print("\nFinal learned weights:", weights)
