# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 08:11:53 2020

@author: jyoth
"""
#pandas for dataframe operations
import pandas as pd

#sklearn for linear regression and metrics
from sklearn import linear_model
from sklearn.metrics import accuracy_score
data=pd.read_csv("C:\\Users\\jyoth\\Documents\\FOLDERS FROM DESKTOP\\mart-sales\\train_v9rqX0R.csv")
print(data.head(5))

#to have a check on the data types for the corresponding columns
print(data.columns)
#find the number of null values in each column 
print(data.isnull().sum())

print(data.dtypes)

print("The attributes which are categorical are- ")
print(list(data.select_dtypes(['object']).columns))


print("Number of unique items",data['Item_Identifier'].nunique())
print("Unique fat content",data['Item_Fat_Content'].nunique())
print("Number of unique item types",data['Item_Type'].nunique())
print("Number of unique outlets",data['Outlet_Identifier'].nunique())
print("No. of unique Outlet sizes",data['Outlet_Size'].nunique())
print("No. of unique Outlet Location types",data['Outlet_Location_Type'].nunique())
print("No. of unique Outlet types",data['Outlet_Type'].nunique())
print("The inference is that we can one hot encode all the columns except Item_Identifier as it has lot of values")
print("This could lead to memory issues so we will deal with an extra step before encoding it")

print(list(data.select_dtypes(['float64']).columns))
print(list(data.select_dtypes(['int64']).columns))


#replace the null values in Item_Weight column by the mean
mean=data['Item_Weight'].mean()
print("Mean of item weight",mean)
data['Item_Weight']=data['Item_Weight'].fillna(mean)

#null values in Outlet_Size will be replaced by the mode Outlet_Size
cols = ["Outlet_Size"]
data[cols]=data[cols].fillna(data.mode().iloc[0])

#to confirm that all columns have all the values filled 
print(data.isnull().sum())

#since one hot encoding for the following column will produce a large number of 
#columns, we consider only the first three letters for item type: assumption that 
#the first three letters contribute to a particular kind of items and the numbers
#are just specific things under that item type. For eg: if diary is DIA, milk could be 
#DIA01, cheese could be DIA02
data['Item_Identifier'] = [x[0:3] for x in data['Item_Identifier']]
print(data['Item_Identifier'].head(4))

#now we find that the number of unique types has drastically reduced
print("Number of unique types of item identifiers",data['Item_Identifier'].nunique())

print("The unique types of fat are: ",data['Item_Fat_Content'].unique())

#cleaning up the data in the following column- different strings for the same value
data["Item_Fat_Content"].replace({"reg": "Regular", "LF" :"Low Fat","low fat":"Low Fat"}, inplace=True)
print("The unique types of fat are: ",data['Item_Fat_Content'].unique())


print("data['Item_Identifier']\n",data['Item_Identifier'].unique())
print("data['Item_Fat_Content']\n",data['Item_Fat_Content'].unique())
print("data['Item_Type']\n",data['Item_Type'].unique())
print("data['Outlet_Identifier']\n",data['Outlet_Identifier'].unique())
print("data['Outlet_Size']\n",data['Outlet_Size'].unique())
print("data['Outlet_Location_Type']\n",data['Outlet_Location_Type'].unique())
print("data['Outlet_Type']\n",data['Outlet_Type'].unique())

print(data.info())
print(data[(data['Outlet_Location_Type']=="Tier 1") & (data['Outlet_Size']=="Medium")].count())
data[(data['Outlet_Location_Type']=="Tier 2") & (data['Outlet_Size']=="")].count()
print(data['Outlet_Location_Type'][5],data['Outlet_Size'][5])
print(data.head(10))
print(data.iloc[2170])
print(data.isnull().sum())

print(data.shape)

one_hot = pd.get_dummies(data['Item_Identifier'])
data = data.drop('Item_Identifier',axis = 1)
data = data.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))

one_hot = pd.get_dummies(data['Item_Fat_Content'])
data = data.drop('Item_Fat_Content',axis = 1)
data = data.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))

one_hot = pd.get_dummies(data['Item_Type'])
data = data.drop('Item_Type',axis = 1)
data = data.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))

one_hot = pd.get_dummies(data['Outlet_Identifier'])
data = data.drop('Outlet_Identifier',axis = 1)
data = data.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


one_hot = pd.get_dummies(data['Outlet_Size'])
data = data.drop('Outlet_Size',axis = 1)
data = data.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


one_hot = pd.get_dummies(data['Outlet_Location_Type'])
data = data.drop('Outlet_Location_Type',axis = 1)
data = data.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


one_hot = pd.get_dummies(data['Outlet_Type'])
data = data.drop('Outlet_Type',axis = 1)
data = data.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))

print(data.shape)
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
lm = linear_model.LinearRegression()

#consists of the independent variables
data_x = data.drop(['Item_Outlet_Sales'],axis=1)
data_x_train = data_x[:-1000]
data_x_test = data_x[-1000:]

#dependent variable
data_y = data['Item_Outlet_Sales']
data_y_train = data_y[:-1000]
data_y_test = data_y[-1000:]

#linear regression
model = lm.fit(data_x_train,data_y_train)
data_y_pred=model.predict(data_x_test)
from sklearn.metrics import mean_squared_error
from math import sqrt
print('Mean squared error: %.2f' % mean_squared_error(data_y_test, data_y_pred))
print(data_y_test)
print(data_y_pred)
print(sqrt(mean_squared_error(data_y_test, data_y_pred)))


##HERE WE BEGIN WITH PROCESSING OF THE TEST FILE
test_data=pd.read_csv("test_AbJTz2l.csv")
test_data.isnull().sum()


"""




"""
#replace the null values in Item_Weight column by the mean
mean=test_data['Item_Weight'].mean()
print("Mean of item weight",mean)
test_data['Item_Weight']=test_data['Item_Weight'].fillna(mean)

#null values in Outlet_Size will be replaced by the mode Outlet_Size
cols = ["Outlet_Size"]
test_data[cols]=test_data[cols].fillna(test_data.mode().iloc[0])

#to confirm that all columns have all the values filled 
print(test_data.isnull().sum())

#since one hot encoding for the following column will produce a large number of 
#columns, we consider only the first three letters for item type: assumption that 
#the first three letters contribute to a particular kind of items and the numbers
#are just specific things under that item type. For eg: if diary is DIA, milk could be 
#DIA01, cheese could be DIA02
test_data['Item_Identifier'] = [x[0:3] for x in test_data['Item_Identifier']]
print(test_data['Item_Identifier'].head(4))

#now we find that the number of unique types has drastically reduced
print("Number of unique types of item identifiers",test_data['Item_Identifier'].nunique())

print("The unique types of fat are: ",test_data['Item_Fat_Content'].unique())

#cleaning up the data in the following column- different strings for the same value
test_data["Item_Fat_Content"].replace({"reg": "Regular", "LF" :"Low Fat","low fat":"Low Fat"}, inplace=True)
print("The unique types of fat are: ",test_data['Item_Fat_Content'].unique())



print(test_data.isnull().sum())

print(test_data.shape)

one_hot = pd.get_dummies(test_data['Item_Identifier'])
test_data = test_data.drop('Item_Identifier',axis = 1)
test_data = test_data.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))

one_hot = pd.get_dummies(test_data['Item_Fat_Content'])
test_data = test_data.drop('Item_Fat_Content',axis = 1)
test_data = test_data.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))

one_hot = pd.get_dummies(test_data['Item_Type'])
test_data = test_data.drop('Item_Type',axis = 1)
test_data = test_data.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))

one_hot = pd.get_dummies(test_data['Outlet_Identifier'])
test_data = test_data.drop('Outlet_Identifier',axis = 1)
test_data = test_data.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


one_hot = pd.get_dummies(test_data['Outlet_Size'])
test_data = test_data.drop('Outlet_Size',axis = 1)
test_data = test_data.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


one_hot = pd.get_dummies(test_data['Outlet_Location_Type'])
test_data = test_data.drop('Outlet_Location_Type',axis = 1)
test_data = test_data.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))


one_hot = pd.get_dummies(test_data['Outlet_Type'])
test_data = test_data.drop('Outlet_Type',axis = 1)
test_data = test_data.join(one_hot)
print(one_hot.columns)
print(one_hot.head(3))

print(test_data.shape)
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
lm = linear_model.LinearRegression()


test_data_x=test_data


#linear regression
model = lm.fit(data_x_train,data_y_train)
test_data_y_pred=model.predict(test_data_x)
test_data_y_pred = test_data_y_pred.where(test_data_y_pred < 0, 0)
submission=pd.read_csv("sample_submission_8RXa3c6.csv")
print(test_data_y_pred.shape)
print(test_data_y_pred)
submission.columns
submission.shape


submission['Item_Outlet_Sales']=test_data_y_pred
submission.head(4)

pd.DataFrame(submission, columns=['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales']).to_csv('sales.csv')
