# 1. Imports
import pandas as pd
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

# 2. Load Dataset
df_flights = pd.read_csv(
    "https://raw.githubusercontent.com/ismayc/pnwflights14/master/data/flights.csv"
)

# 3. Select Categorical Columns
cat_df_flights = df_flights.select_dtypes(include=['object']).copy()

# 4. Handle Missing Values
print("Missing Values Before:", cat_df_flights.isnull().values.sum())
cat_df_flights = cat_df_flights.fillna(
    cat_df_flights['tailnum'].value_counts().index[0]
)
print("Missing Values After:", cat_df_flights.isnull().values.sum())

# -------------------------
# PART 1: Replace Values
# -------------------------
# Manual replace map
replace_map = {
    'carrier': {
        'AA': 1, 'AS': 2, 'B6': 3, 'DL': 4, 'F9': 5,
        'HA': 6, 'OO': 7, 'UA': 8, 'US': 9, 'VX': 10, 'WN': 11
    }
}

# Generate mapping automatically
labels = cat_df_flights['carrier'].astype('category').cat.categories.tolist()
replace_map_comp = {
    'carrier': {k: v for k, v in zip(labels, list(range(1, len(labels) + 1)))}
}
print("Generated Replace Map:\n", replace_map_comp)

# Apply replacement
cat_df_flights_replace = cat_df_flights.copy()
cat_df_flights_replace.replace(replace_map_comp, inplace=True)

print("Carrier Column After Replacement:\n", cat_df_flights_replace[['carrier']].head())
print("Carrier dtype:", cat_df_flights_replace['carrier'].dtypes)

# Convert object -> category for efficiency
cat_df_flights_lc = cat_df_flights.copy()
cat_df_flights_lc['carrier'] = cat_df_flights_lc['carrier'].astype('category')
cat_df_flights_lc['origin'] = cat_df_flights_lc['origin'].astype('category')

print("Categorical Conversion:\n", cat_df_flights_lc.dtypes)

# Performance check
print("Object dtype groupby timing:")
%timeit cat_df_flights.groupby(['origin', 'carrier']).count()
print("Category dtype groupby timing:")
%timeit cat_df_flights_lc.groupby(['origin', 'carrier']).count()

# -------------------------
# PART 2: Label Encoding
# -------------------------
# Using pandas categorical codes
cat_df_flights_lc['carrier'] = cat_df_flights_lc['carrier'].cat.codes
print("Carrier after Label Encoding (pandas):\n", cat_df_flights_lc[['carrier']].head())

# Specific encoding using np.where
cat_df_flights_specific = cat_df_flights.copy()
cat_df_flights_specific['US_code'] = np.where(
    cat_df_flights_specific['carrier'].str.contains('US'), 1, 0
)
print(cat_df_flights_specific[['carrier', 'US_code']].head())

# Using sklearn LabelEncoder
cat_df_flights_sklearn = cat_df_flights.copy()
lb_make = LabelEncoder()
cat_df_flights_sklearn['carrier_code'] = lb_make.fit_transform(cat_df_flights['carrier'])
print(cat_df_flights_sklearn[['carrier', 'carrier_code']].head())

# -------------------------
# PART 3: One-Hot Encoding
# -------------------------
# Using pandas get_dummies
cat_df_flights_onehot = cat_df_flights.copy()
cat_df_flights_onehot = pd.get_dummies(
    cat_df_flights_onehot, columns=['carrier'], prefix=['carrier']
)
print("Pandas One-Hot:\n", cat_df_flights_onehot.head())

# Using sklearn LabelBinarizer
cat_df_flights_onehot_sklearn = cat_df_flights.copy()
lb = LabelBinarizer()
lb_results = lb.fit_transform(cat_df_flights_onehot_sklearn['carrier'])
lb_results_df = pd.DataFrame(lb_results, columns=lb.classes_)

# Combine original + encoded
result_df = pd.concat([cat_df_flights_onehot_sklearn, lb_results_df], axis=1)
print("Sklearn One-Hot:\n", result_df.head())
