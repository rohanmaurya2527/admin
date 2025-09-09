# 1. Imports
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 2. Load Dataset
# Option A: CSV
#df = pd.read_csv("your_dataset.csv")


from sklearn.datasets import load_diabetes
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["Target"] = data.target

# 3. Basic Exploration
print("Shape:", df.shape)
print("Info:")
df.info()
print("Description:", df.describe())
print("Missing Values Count:", df.isnull().sum())
print("Missing Values Percentage:", df.isnull().mean() * 100)

# 4. Define Columns for Imputation
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
print("Numeric Columns Available for Imputation:", numeric_cols)

# Example: pick any numeric column (replace with your column name)
col = "age" if "age" in df.columns else numeric_cols[0]

# 5. Imputation
mean_val = df[col].mean()
median_val = df[col].median()
mode_val = df[col].mode()[0]

df[f"{col}_mean"] = df[col].fillna(mean_val)
df[f"{col}_median"] = df[col].fillna(median_val)
df[f"{col}_mode"] = df[col].fillna(mode_val)

# 6. Visualization
df[col].plot(kind='kde', label=f"Original {col}")
df[f"{col}_mean"].plot(kind='kde', color='red', label="Mean Imputation")
df[f"{col}_median"].plot(kind='kde', color='yellow', label="Median Imputation")
df[f"{col}_mode"].plot(kind='kde', color='green', label="Mode Imputation")

plt.legend()
plt.title(f"{col} Distribution - Original vs Imputation")
plt.show()
