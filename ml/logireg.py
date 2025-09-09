import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset and drop missing values
df = pd.read_csv("Datasets/Car_Insurance_Claim.csv")[["CREDIT_SCORE", "OUTCOME"]].dropna()

# Features and target
x,y = df[["CREDIT_SCORE"]],df["OUTCOME"]

# Split into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)

# Train logistic regression model
model = LogisticRegression()
model.fit(xtrain, ytrain)

# Plot: scatter of all data + logistic regression on training data
plt.figure(figsize=(8, 5))
plt.scatter(x, y, alpha=0.5, label="Data")
sns.regplot(x=xtrain, y=ytrain, logistic=True, scatter=False, color="red", label="Logistic Fit")
plt.xlabel("CREDIT_SCORE")
plt.ylabel("OUTCOME")
plt.title("Logistic Regression: CREDIT_SCORE vs OUTCOME")
plt.legend()
plt.show()

# Predict and evaluate
y_pred = model.predict(xtest)
print("Accuracy:", accuracy_score(ytest, y_pred))
print("Confusion Matrix:", confusion_matrix(ytest, y_pred))
