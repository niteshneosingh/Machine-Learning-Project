# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 11:49:53 2018

@author: nitesh
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the datasets
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values  #independent variable 
Y = dataset.iloc[:,4].values #Dependent variable Profit

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3]) #we want 4th column to encode into numbers like 0,1,2...
onehotencoder = OneHotEncoder(categorical_features = [3]) #creates dummy variables
X = onehotencoder.fit_transform(X).toarray()

#splitting training set & test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

#avoiding the dummy variable trap
X = X[:, 1:]

#Creating Training & test set (40 records - training, 10 records - test)
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#fitting multiple linear regression model to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train) 

#Predicting the tet set result
Y_pred = regressor.predict(X_test)

#Building the optimal model using backward elimination
import statsmodels.formula.api as sm  # but it ignores the coefficient with no independent variables
X = np.append(arr = np.ones((50,1)).astype(int),values = X, axis =1) #appending array X with a column of 1

X_opt = X[:,[0,1,2,3,4,5]] # inserting every independent variable 
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit() #filling regressor model with X_opt & Y Step - 2
regressor_OLS.summary() #Step - 3 to know the p-value

""" x2 has highest value so removing it"""
X_opt = X[:,[0,1,3,4,5]] # inserting every independent variable 
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit() #filling regressor model with X_opt & Y Step - 2
regressor_OLS.summary() #Step - 3 to know the p-value

""" x1 has next highest p value which is greater than 0.05(5%)"""
X_opt = X[:,[0,3,4,5]] # inserting every independent variable 
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit() #filling regressor model with X_opt & Y Step - 2
regressor_OLS.summary()

""" x5 has next highest p value which is greater than 0.05(5%)"""
X_opt = X[:,[0,3,4]] # inserting every independent variable 
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit() #filling regressor model with X_opt & Y Step - 2
regressor_OLS.summary()

""" x4 has next highest p value which is greater than 0.05(5%)"""
X_opt = X[:,[0,3]] # inserting every independent variable 
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit() #filling regressor model with X_opt & Y Step - 2
regressor_OLS.summary()

""" CONCLUSION - PROFIT DEPENDS HEAVILY ON R&D EXPENDITURE"""
