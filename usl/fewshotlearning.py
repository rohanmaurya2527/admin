# Practical 3
# Aim: Demonstration of Few Shot Learning

from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target
classes = data.target_names

# Display column information
print("Total Columns:", len(data.feature_names))
print("\nColumn Names:")
for name in data.feature_names:
    print("-", name)

# Show sample data
df = pd.DataFrame(X, columns=data.feature_names)
print("\nSample Data:")
print(df.head())

# Select Few-Shot samples (3 from each class)
X_few = []
y_few = []

for cls in np.unique(y):
    idx = np.where(y == cls)[0][:3]
    X_few.append(X[idx])
    y_few.append(y[idx])

X_few = np.vstack(X_few)
y_few = np.hstack(y_few)

print("\nFew-Shot Samples Used:", len(X_few))

# Train model
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_few, y_few)

# Test on full dataset
pred_all = model.predict(X)
print("\nAccuracy on Full Dataset:", accuracy_score(y, pred_all))

# Prediction on existing dataset sample
sample = X[451].reshape(1, -1)
pred = model.predict(sample)[0]
print("\nPrediction on dataset sample (X[451]):", classes[pred])

# Prediction on manual input
new_data = np.array([
14.5, 20.1, 95.0, 600.0, 0.11, 0.12, 0.09, 0.06, 0.20, 0.07,
0.20, 1.10, 1.30, 8.0, 0.006, 0.01, 0.02, 0.007, 0.015, 0.003,
16.0, 25.0, 110.0, 800.0, 0.14, 0.18, 0.15, 0.08, 0.27, 0.09
]).reshape(1, -1)

pred2 = model.predict(new_data)[0]
print("\nPrediction on Manual Input:", classes[pred2])
