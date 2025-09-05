# Step 1: Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Step 2: Load a sample dataset (Iris dataset for classification)
iris = datasets.load_iris()
X = iris.data[:, :2]  # Using only two features for visualization
y = iris.target

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train the SVM model
svm_model = SVC(kernel='linear')  # Using linear kernel
svm_model.fit(X_train, y_train)

# Step 5: Make predictions and evaluate the model
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy of the SVM model: {accuracy * 100:.2f}%')

# Plotting decision boundary
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', marker='o', s=100, edgecolors='k')
plt.title("SVM Decision Boundary")

# Create grid for decision boundary
xx, yy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 100),
                     np.linspace(X[:, 1].min(), X[:, 1].max(), 100))

# Get decision boundary
Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.show()
