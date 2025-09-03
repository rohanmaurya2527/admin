import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv("Rizwan/Datasets/Ice Cream.csv")
"""
Using predefined data then following lines by skipping above lines
from sklearn.datasets import load_diabetes
data = load_diabetes()
"""
# Features and target
x = df[["Temperature"]]  # 2D array
y = df[["Revenue"]]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=25)

# Create and train the model
model = LinearRegression()
model.fit(x_train, y_train)

# Predict revenue for temperature = 33
predicted_revenue = model.predict([[33]])#m=model.coef_, c=model.intercept_
print("Predicted Revenue for Temperature 33°C:", predicted_revenue[0][0])

# Plotting
plt.scatter(df["Temperature"], df["Revenue"], color='blue', label='Data points')
plt.plot(x_train, model.predict(x_train), color='red', label='Regression line')
plt.xlabel("Temperature")
plt.ylabel("Revenue")
plt.title("Ice Cream Revenue vs Temperature")
plt.legend()
plt.show()
