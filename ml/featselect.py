# 1. Imports
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.feature_selection import VarianceThreshold, chi2

# 2. Load Dataset
data = load_diabetes()

# 3. Prepare Features (X) and Target (y)
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.DataFrame(data.target, columns=['target'])

# 4. Combine into Single DataFrame
df = pd.concat([X, y], axis=1)

# 5. Correlation Analysis
plt.figure(figsize=(10, 8))
sb.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 6. Variance Threshold Feature Selection
selector = VarianceThreshold(threshold=0.2)
X_var = selector.fit_transform(X)
selected_features = X.columns[selector.get_support()]
print("Selected Features (Variance Threshold):", selected_features.tolist())

# 7. Chi-Square Test
chi2_scores, p_values = chi2(X, y)
chi2_results = pd.DataFrame({
    "Feature": X.columns,
    "Chi2 Score": chi2_scores,
    "p-value": p_values
})
print("Chi-Square Results:\n", chi2_results)
