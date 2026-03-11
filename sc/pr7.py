import numpy as np
def delta_rule_learning(X, t, lr=0.1, epochs=5):
    weights = np.zeros(X.shape[1])
    print("Initial Weights:", weights)
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}")
        for i, (x_i, target) in enumerate(zip(X, t), 1):
            y = np.dot(x_i, weights)
            error = target - y
            delta_w = lr * error * x_i
            weights += delta_w
            # Print details
            print(f"Sample {i}: Input={x_i}, Target={target}, Output={y:.2f}, Error={error:.2f}")
            print("Weight Update:", delta_w)
            print("New Weights:", weights)
    return weights
# Dataset
X = np.array([[1, 2],[2, 1],[3, 4],[4, 3]])
t = np.array([1, 1, -1, -1])
# Train network
final_weights = delta_rule_learning(X, t)
print("\nFinal Learned Weights:", final_weights)
