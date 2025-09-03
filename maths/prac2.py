import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("sales_data - sales_data.csv")
df.head()

df.describe()

df.isnull().sum()

