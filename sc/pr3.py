import numpy as np
class BAM:
    def __init__(self, x_neurons, y_neurons):
        self.W = np.zeros((x_neurons, y_neurons))
    def train(self, patterns):
        for x, y in patterns:
            self.W += np.outer(x, y)
    def recall_x_to_y(self, x):
        return np.where(np.dot(x, self.W) >= 0, 1, -1)
    def recall_y_to_x(self, y):
        return np.where(np.dot(y, self.W.T) >= 0, 1, -1)
x1 = np.array([1, 1, 1, -1])
y1 = np.array([1, 1])
x2 = np.array([-1, -1, 1, 1])
y2 = np.array([-1, 1])
bam = BAM(4, 2)
bam.train([(x1, y1), (x2, y2)])
print("Weights:\n", bam.W)
input_x = np.array([1, 1, 1, -1])
input_y = np.array([-1, 1])
print(f"Input X: {input_x}")
print("Recalled Y:", bam.recall_x_to_y(input_x))
print(f"\nInput Y: {input_y}")
print("Recalled X:", bam.recall_y_to_x(input_y))
noisy_x = np.array([1, -1, 1, -1])
print(f"Noisy Input X: {noisy_x}")
print("Recalled Y from noisy X:", bam.recall_x_to_y(noisy_x))
