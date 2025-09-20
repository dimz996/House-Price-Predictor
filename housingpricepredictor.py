# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 23:44:43 2025

@author: Dimuthu Fernando
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv(r"Housing.csv")

X = df [["area","bedrooms","bathrooms","stories","parking"]].values
y = df["price"].values


#spiliting data 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state =42)
X_train.shape,X_test.shape 


#Normalizing the data to fit 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)

#initilizing parameters for the gradient descent
m,n = X_train.shape
w = np.zeros(n)
b = 0
learning_rate = 0.01
iterations =  1000

#gradient descent
for i in range(iterations):
    y_pred = np.dot(X_train, w) + b
    error = y_pred - y_train
    
    dw = (2/m)*np.dot(X_train.T,error)
    db = (2/m) * np.sum(error)
    
    w -= learning_rate * dw
    b -= learning_rate * db
    
    if i % 100 == 0:
        cost = (1/m) * np.sum (error **2 )
        print(f"Iteration {i}, cost: {cost:.2f}")
        
print("Learned Weights:",w)
print("Learned Bias:",b)

y_test_pred = np.dot(X_test, w) +b
print("First 5 Predictions:", y_test_pred[:5])
print("First 5 Actual:", y_test[:5])


    
    
    






