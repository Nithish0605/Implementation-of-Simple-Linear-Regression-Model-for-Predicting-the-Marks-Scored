# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries. 
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```PYTHON
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: NITHISH S
RegisterNumber:  212224240105
*/import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
DATASET
![Screenshot 2025-03-15 131005](https://github.com/user-attachments/assets/19d0325b-115f-40d0-8c40-ff06924be466)

HEAD VALUES :
![Screenshot 2025-03-15 131022](https://github.com/user-attachments/assets/21a0f62a-fda3-42b5-a03c-61fd598218b1)

TAIL VALUES:
![Screenshot 2025-03-15 131033](https://github.com/user-attachments/assets/d2666ecc-b7f0-4775-b83d-90f73060191a)

X AND Y VALUES
![Screenshot 2025-03-15 131054](https://github.com/user-attachments/assets/08c3138f-5128-4b6b-ae34-5228bd831155)

PREDICTION VALUES OF X AND Y :
![Screenshot 2025-03-15 131110](https://github.com/user-attachments/assets/060f8094-6e70-47ad-9221-83b92064ca1b)

MSE,MAE,RMSE VALUES :
![Screenshot 2025-03-15 131208](https://github.com/user-attachments/assets/e974e5fc-dcef-4b22-aa5b-96105a6c7eb7)

TRAINING SET 
![Screenshot 2025-03-15 131120](https://github.com/user-attachments/assets/b6367a3b-d27e-4175-8504-43b69e483dab)

TESTING SET:
![Screenshot 2025-03-15 131144](https://github.com/user-attachments/assets/9ec6f1df-b321-437c-9f1b-f7040e00d454)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
