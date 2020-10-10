# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 11:01:27 2020

@author: jyoth
"""

import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("train_yaOffsB.csv")
test=pd.read_csv("test_pFkWwen.csv")

print(data.isnull().sum())
print(test.isnull().sum())


data=data.drop('Number_Weeks_Used',axis=1)
test=test.drop('Number_Weeks_Used',axis=1)

"""since number_weeks_used is the only attribute with missing values,
we try to get results without the value and see how accurate it is
though this is not the right way, this is just to check the influence
of all parameters"""

attributes=list(data.columns)
print(attributes)
for col in attributes:
    print(col)
    print(data[col].unique())
    print("\n\n")
    

from scipy.stats import chi2_contingency 

independent=[]
dependent=[]
#Crop damage and season
for col in attributes:
    if col!='Crop_Damage':
        data_crosstab = pd.crosstab(data.Crop_Damage, data[col],  margins = False) 
        print(data_crosstab) 
        stat, p, dof, expected = chi2_contingency(data_crosstab) 
           # interpret p-value 
        alpha = 0.05
        print("p value is " + str(p)) 
        if p <= alpha: 
            dependent.append(col)
        else: 
            independent.append(col)


print("Independent",independent)
data=data.drop(["ID","Season"],axis=1)
test=test.drop(["ID","Season"],axis=1)

print(data.dtypes)

print("We have to one hot encode Pesticide_Use_Category")
one_hot = pd.get_dummies(data['Pesticide_Use_Category'])
data = data.drop('Pesticide_Use_Category',axis = 1)
data = data.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


one_hot = pd.get_dummies(test['Pesticide_Use_Category'])
test = test.drop('Pesticide_Use_Category',axis = 1)
test = test.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))

#data
data=data.rename(columns={1: "PesticideCategory1",2: "PesticideCategory3",3: "PesticideCategory3"},errors='raise')
print(data.columns)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
#sc = StandardScaler()
X_train=data.drop('Crop_Damage',axis=1)
y_train=data['Crop_Damage']
X_test=test
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

import numpy as np
y_pred=clf.predict(X_test)
print(np.unique(y_pred))

submission=pd.read_csv("sample_submission_O1oDc4H.csv")
submission['Crop_Damage']=y_pred
submission.head(4)
print(submission['Crop_Damage'].unique())
print(data['Crop_Damage'].unique())

pd.DataFrame(submission, columns=['ID','Crop_Damage']).to_csv('agriculture.csv')
