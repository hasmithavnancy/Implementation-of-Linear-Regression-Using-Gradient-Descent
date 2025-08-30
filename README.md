# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.add a column to x for the intercept,initialize the theta
2.perform graadient descent
3.read the csv file
4.assuming the last column is ur target variable 'y' and the preceeding column
5.learn model parameters
6.predict target value for a new data point

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: HASMITHA V NANCY
RegisterNumber: 212224040111 
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1, y, learning_rate=0.1, num_iters=1000):
    X = np.c_[np.ones(len(X1)), X1]  
    theta = np.zeros((X.shape[1], 1))  
    
    for _ in range(num_iters):
        predictions = X.dot(theta)  
        errors = predictions - y   
        theta -= learning_rate * (1/len(X1)) * X.T.dot(errors)  
    
    return theta

data = pd.read_csv("C:/Users/admin/Downloads/50_Startups.csv", header=None)
data.head()

X = data.iloc[1:, :-2].values  
X1 = X.astype(float)
y = data.iloc[1:, -1].values.reshape(-1, 1)  # Target

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X1_Scaled = scaler_X.fit_transform(X1)
Y1_Scaled = scaler_y.fit_transform(y)

print("Original X:\n", X)
print("Scaled X:\n", X1_Scaled)

theta = linear_regression(X1_Scaled, Y1_Scaled)

new_data = np.array([[165349.2, 136897.8, 471784.1]])  
new_Scaled = scaler_X.transform(new_data)  
prediction = np.dot(np.append([1], new_Scaled), theta).reshape(-1, 1)

pre = scaler_y.inverse_transform(prediction)
print("Prediction (scaled):", prediction)
print(f"Predicted value: {pre}")

```

## Output:
<img width="572" height="819" alt="Screenshot 2025-08-30 141313" src="https://github.com/user-attachments/assets/68d9948f-54ff-42db-91d4-834ac568bd47" />
<img width="580" height="819" alt="Screenshot 2025-08-30 141327" src="https://github.com/user-attachments/assets/cf28102f-eea5-4967-83be-838bad01c48d" />
<img width="543" height="602" alt="Screenshot 2025-08-30 141342" src="https://github.com/user-attachments/assets/a6fa916c-98e9-4df4-b943-9def7ac2e0f4" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
