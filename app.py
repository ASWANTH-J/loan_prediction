# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 19:30:51 2021

@author: Mithula Roy 
"""

import numpy as np
from flask import Flask, request, render_template
import pickle
import pandas as pd
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

#for creating emi feature
def emi_calculator(x):
  p=x[0]
  n=x[1]
  r = 0.09 / 12.0
  a=(1+r)**n
  emi = (p*r) * (a/(a-1))
  return np.round(emi*1000,2)

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    vals =request.form.values()
    lst = [val for val in vals]
    dic = {
            'Gender' : lst[1],
            'Married' : lst[2],
            'Dependents' : lst[3],
            'Education' : lst[4],
            'Self_Employed' : lst[5],
            'ApplicantIncome' : int(lst[6]),
            'CoapplicantIncome' : float(lst[7]),
            'LoanAmount' : float(lst[8]),
            'Loan_Amount_Term' : float(lst[9]),
            'Property_Area' :   lst[11]
        }
    cr = int(lst[10])                    
    df = pd.DataFrame(dic,index=[0])
    df['TotalIncome'] = df['ApplicantIncome']+df['CoapplicantIncome']
    df['Credit_History_0.0'] = 0 if cr else 1
    df['Credit_History_1.0'] = 1 if cr else 0
    df['AmountPerIncome'] = (df['LoanAmount'] * 1000) / df['TotalIncome']
    df['emi'] = df[['LoanAmount','Loan_Amount_Term']].apply(emi_calculator,axis=1)
    
    #Encoding
    #encoding values
    df['Gender'] = df['Gender'].map({'Male':1,'Female':0})
    df['Dependents'] = df['Dependents'].map({'0':0,'1':1,'2':2,'3+':3})
    df['Married'] = df['Married'].map({'Yes':1,'No':0})
    df['Education'] = df['Education'].map({'Graduate':1,'Not Graduate':0})  
    df['Property_Area'] = df['Property_Area'].map({'Rural':1,'Urban':2,'Semiurban':3})
    df['Self_Employed'] = df['Self_Employed'].map({'Yes':1,'No':0})
    
    pred = model.predict(df)[0]
    if pred==1:
        txt = 'You\'re eligible for loan'
    else:
        txt = 'You\'re not eligible for loan'
    
    return render_template('result.html', prediction_text=txt)



if __name__ == "__main__":
    app.run(debug=True)