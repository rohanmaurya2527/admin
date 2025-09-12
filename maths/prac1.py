import pandas as pd
import warnings as war
war.filterwarnings('ignore')
# Load dataset (assuming first column is an index)
df = pd.read_csv("risk_analytics_train - risk_analytics_train.csv", index_col=0)

# Basic exploration
print(df.head())          # Show first 5 rows
print(df.dtypes)          # Data types of columns
print(df.shape)           # Shape of the dataset (rows, columns)
print(df.columns)         # Column names
print(df.isnull().sum())  # Count missing values in each column
print(df.describe(include='all'))  # Summary statistics (categorical + numeric)

# Fill missing values for categorical columns using the most frequent value (mode)
categorical_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term']
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Check missing values again
print(df.isnull().sum())

# Fill missing 'LoanAmount' with rounded mean
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)

# Fill missing 'Credit_History' with 0
df['Credit_History'].fillna(0, inplace=True)

# Final check for any remaining missing values
print(df.isnull().sum())
