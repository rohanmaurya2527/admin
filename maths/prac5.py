import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
df=pd.read_csv("sales_data - sales_data.csv")
df.head()

df.iloc[:,5:12].nunique()

df.describe()

categ_counts=df['Country'].value_counts()
categ_counts

#Bar Plots
sn.countplot(data=df,x='Customer_Gender')
plt.title('Categorical Data Disstribution')

#Histogram
plt.hist(df['Month'],bins=10,edgecolor='black')
plt.title('Revenue Histogram')
plt.xlabel('Month')
plt.ylabel('Revenue')
plt.legend()

#One Hot Encoder
one_hot_encoded=pd.get_dummies(df,columns=['Product'])
one_hot_encoded

one_hot_encoded=pd.get_dummies(df,columns=[State])
one_hot_encoded
