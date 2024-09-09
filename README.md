# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1 - Start

STEP 2 -Import the necessary python packages

STEP 3 - Read the dataset.

STEP 4 - Define X and Y array.

STEP 5 - Define a function for costFunction,cost and gradient.

STEP 6- Define a function to plot the decision boundary and predict the Regression value

STEP 7- End

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: S LALIT CHANDRAN
RegisterNumber:  212223240077
*/
```
```
import pandas as pd
import numpy as np
data=pd.read_csv("/content/Placement_Data (1).csv")
data.head()
data1=data.copy()
data1.head()
data1=data.drop(['sl_no','salary'],axis=1)
data1
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
X=data1.iloc[:,: -1]
Y=data1["status"]
theta=np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
  return 1/(1+np.exp(-z))
def loss(theta,X,y):
 h=sigmoid(X.dot(theta))
  return -np.sum(y*np.log(h)+ (1-y) * np.log(1-h))
def gradient_descent(theta,X,y,alpha,num_iterations):
  m=len(y)
  for i in range(num_iterations):
    h=sigmoid(X.dot(theta))
    gradient=X.T.dot(h-y)/m
    theta-=alpha*gradient
  return theta
theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)
def predict(theta,X):
  h=sigmoid(X.dot(theta))
  y_pred=np.where(h>=0.5 , 1,0)
  return y_pred
y_pred=predict(theta,X)
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
print("Predicted:\n",y_pred)
print("Actual:\n",y.values)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print("Predicted Result:",y_prednew)

```

## Output:


Accuracy

![WhatsApp Image 2024-09-06 at 11 36 16_d89a7d82](https://github.com/user-attachments/assets/7d5f084c-5c1b-4ce8-9bc2-ddd74194b855)


Y predict

![WhatsApp Image 2024-09-06 at 11 36 28_0907192b](https://github.com/user-attachments/assets/291d17f7-a0aa-4a74-8911-49bbd26ba4fc)


Y

![WhatsApp Image 2024-09-06 at 11 36 38_9fed8e00](https://github.com/user-attachments/assets/68f82149-6526-4159-a8bf-00fb3874a997)


![WhatsApp Image 2024-09-06 at 11 36 57_550a1311](https://github.com/user-attachments/assets/98accaa0-6462-44bb-aac3-b626bbbf0e34)




## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

