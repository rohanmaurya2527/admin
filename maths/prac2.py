import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("sales_data - sales_data.csv")
df.head()

df.describe()

df.isnull().sum()

#DATA VISUALIZATION
#1. Sales Trend Over Time 
df['Date']=pd.to_datetime(df['Date'])
daily_sale=df.groupby('Date')['Revenue'].sum()
plt.figure(figsize=(10,5))
plt.plot(daily_sale.index,daily_sale.values)
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.title('Sales Trend Over Time')
plt.grid(True)
plt.show()

#2. Visualizing Sales by Product Category
categ_sales=df.groupby('Product_Category')['Revenue'].sum()
plt.figure(figsize=(10,6))
categ_sales.plot(kind='bar')
plt.xlabel('Product Category')
plt.ylabel('Total Sales')
plt.title('Sales by Product Category')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

#3. Visualizing Sales Distribution
plt.figure(figsize=(10,6))
plt.hist(df['Revenue'],bins=20)
plt.xlabel('Sales Amount')
plt.ylabel('Freuency')
plt.title('Sales Distribution')
plt.grid(axis='y')
plt.show()



