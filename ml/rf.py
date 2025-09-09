# Import necessary libraries
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on test data
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy:.2f}")
import matplotlib.pyplot as plt
import numpy as np

# === 1. Plot Confusion Matrix ===
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)

disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# === 2. Plot Top 10 Feature Importances ===

# Get feature importances from the trained Random Forest
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]  # Sort feature importances descending

# Plot top 10 features
top_n = 10
plt.figure(figsize=(10, 5))
plt.title("Top 10 Important Features")
plt.barh(range(top_n), importances[indices[:top_n]][::-1], align='center')
plt.yticks(range(top_n), [data.feature_names[i] for i in indices[:top_n]][::-1])
plt.xlabel("Feature Importance Score")
plt.tight_layout()
plt.show()
