import pandas as pd
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

# 1. Load Dataset
df = pd.read_csv("https://raw.githubusercontent.com/ismayc/pnwflights14/master/data/flights.csv")

# 2. Select Categorical Columns
df_cat = df.select_dtypes(include='object').copy()

# 3. Handle Missing Values (e.g., 'tailnum')
df_cat['tailnum'].fillna(df_cat['tailnum'].mode()[0], inplace=True)

# ----------------------------
# 4. Label Encoding
# ----------------------------

# (a) Pandas Categorical Codes
df_cat['carrier_catcode'] = df_cat['carrier'].astype('category').cat.codes

# (b) Sklearn LabelEncoder
le = LabelEncoder()
df_cat['carrier_label'] = le.fit_transform(df_cat['carrier'])

# ----------------------------
# 5. One-Hot Encoding
# ----------------------------

# (a) Using pandas get_dummies
df_onehot_pandas = pd.get_dummies(df_cat['carrier'], prefix='carrier')

# (b) Using sklearn LabelBinarizer
lb = LabelBinarizer()
onehot_result = lb.fit_transform(df_cat['carrier'])
df_onehot_sklearn = pd.DataFrame(onehot_result, columns=lb.classes_)

# ----------------------------
# 6. Final Output (Preview)
# ----------------------------
print("Original + Label Encoded Columns:")
print(df_cat[['carrier', 'carrier_catcode', 'carrier_label']].head())

print("\nOne-Hot Encoding (pandas):")
print(df_onehot_pandas.head())

print("\nOne-Hot Encoding (sklearn):")
print(df_onehot_sklearn.head())
