import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
df=pd.read_csv("Datasets/risk_analytics_train - risk_analytics_train.csv",index_col=0)
df.head()

df.dtypes

df.shape

df.columns

df.isnull().sum()

df.describe(include='all')

for x in ['Gender','Married','Dependents','Self_Employed','Loan_Amount_Term']:
    df[x].fillna(df[x].mode()[0],inplace=True)
df.isnull().sum()

risk['LoanAmount'].fillna(round(risk['LoanAmount'].mean(),0),inplace=True)
risk.isnull().sum()

df.Credit_History.mode()

risk['Credit_History'].fillna(value=0,inplace=True)
risk.isnull().sum()
