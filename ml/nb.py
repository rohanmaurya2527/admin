import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('Datasets/synthetic_text_data.csv')
X, y = data['text'], data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model training and prediction
model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

# Accuracy and confusion matrix
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
cm = confusion_matrix(y_test, y_pred)

# Confusion matrix heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Predict user input
user_input = "I love artificial intelligence and machine learning"
user_vec = vectorizer.transform([user_input])
prediction = model.predict(user_vec)[0]
print(f"Prediction: {prediction}")
