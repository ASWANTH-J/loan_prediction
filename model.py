# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 14:14:44 2021

@author: Aswanth J
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
train = pd.read_csv('dataset/dream_house_loan.csv')

ID,TARGET_COL = 'Loan_ID','Loan_Status'
features = [col for col in train.columns if col not in [ID,TARGET_COL]]

#helper function

#function to impute missing values in the features LoanAmount
def impute_missing_loan_amount(df,features):
  data=df.copy()
  cat_cols = ['Gender', 'Married','Dependents','Education','Self_Employed','Credit_History','Property_Area']
  for col in list(filter(lambda i :i in features,cat_cols)) :
    le = LabelEncoder()
    data[col] = le.fit_transform(data.loc[:,col])
  m_train = data[~data.LoanAmount.isnull()]
  m_test = data[data.LoanAmount.isnull()]
  regressor = LinearRegression()
  regressor.fit(m_train[features],m_train['LoanAmount'])
  y_pred = pd.Series(data=regressor.predict(m_test[features]),index=m_test.index)
  return y_pred

#function for calculating emi
def emi_calculator(x):
  p=x[0]
  n=x[1]
  r = 0.09 / 12.0
  a=(1+r)**n
  emi = (p*r) * (a/(a-1))
  return np.round(emi*1000,2)



#preparing data for the model

#missing value imputation
#Adding a TotalIncome column
train['TotalIncome'] = train['ApplicantIncome']+train['CoapplicantIncome']
features.append('TotalIncome')

#imputing missing values in categorical features
train['Gender'] = train['Gender'].fillna(train.Gender.mode()[0])
train.Married = train.Married.fillna(train.Married.mode()[0])
train.Dependents = train.Dependents.fillna(train.Dependents.mode()[0])
train.Self_Employed = train.Self_Employed.fillna(train.Self_Employed.mode()[0])
train.Loan_Amount_Term = train.Loan_Amount_Term.fillna(train.Loan_Amount_Term.mode()[0])

#imputing missing values in numerical features
y_pred = np.round(impute_missing_loan_amount(train,['TotalIncome','Education']),0)
train.loc[y_pred.index,'LoanAmount'] = y_pred

#encoding values
train['Loan_Status'] = train['Loan_Status'].map({'Y':1,'N':0})
train['Gender'] = train['Gender'].map({'Male':1,'Female':0})
train['Dependents'] = train['Dependents'].map({'0':0,'1':1,'2':2,'3+':3})
train['Married'] = train['Married'].map({'Yes':1,'No':0})
train['Education'] = train['Education'].map({'Graduate':1,'Not Graduate':0})
train['Property_Area'] = train['Property_Area'].map({'Rural':1,'Urban':2,'Semiurban':3})
train['Self_Employed'] = train['Self_Employed'].map({'Yes':1,'No':0})
train = pd.get_dummies(train,columns=['Credit_History'])

#adding new features to data
train['AmountPerIncome'] = (train['LoanAmount'] * 1000) / train['TotalIncome']
train['emi'] = train[['LoanAmount','Loan_Amount_Term']].apply(emi_calculator,axis=1)
features = [col for col in train.columns if col not in [ID,TARGET_COL]]

features
#modeling
tree = DecisionTreeClassifier(min_samples_split=20,max_leaf_nodes=2**3,random_state=42)
_=tree.fit(train[features],train[TARGET_COL])

# Saving model to disk
pickle.dump(tree, open('model.pkl','wb'))