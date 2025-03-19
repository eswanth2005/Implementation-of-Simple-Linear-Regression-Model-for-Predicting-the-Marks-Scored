# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import all the necessary libraries.
2. Read the dataset and extract the feature and target variable
3. Split the dataset into training and testing sets
4. Train the Linear Regression model and make predictions
5. Calculate Error metrics
6. Visualise the training set results

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Eswanth Kumar K
RegisterNumber:  212223040046
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('/content/student_scores.csv')
data.head()
data.tail()
x = data.iloc[:,:-1].values
y = data.iloc[:,1].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size = 1/3, random_state = 42)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
from sklearn.metrics import mean_absolute_error,mean_squared_error
mse = mean_squared_error(y_test,y_pred)
print("Mean Square Error: ", mse)
mae = mean_absolute_error(y_test,y_pred)
print("Mean Absolute Error: ",mae)
rmse = np.sqrt(mse)
print("Root Mean Square Error: ",rmse)
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Hours vs Scores")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

## Output:

## Dataset
![Screenshot 2025-03-19 085955](https://github.com/user-attachments/assets/b3b7f7ea-635d-4c10-9a1b-af92ade2b8ef)

## df.head()
![Screenshot 2025-03-19 090117](https://github.com/user-attachments/assets/be718ee4-9ec6-431c-8fba-ea3a4df146f9)

## df.tail()
![Screenshot 2025-03-19 090247](https://github.com/user-attachments/assets/475bd61c-a9e6-4cdf-9f4a-dd5186fcc0e2)

## X and y
![Screenshot 2025-03-19 091223](https://github.com/user-attachments/assets/2ec8bada-72fa-44c1-a5ea-a0a644d39dbd)

## Y-prediction values
![Screenshot 2025-03-19 090745](https://github.com/user-attachments/assets/be1f0ee0-dd4b-4d44-85fd-6b14ac3c5eae)

## Training set
![Screenshot 2025-03-19 090757](https://github.com/user-attachments/assets/88c67639-5634-4d67-875d-a0c139ff6a70)

## Testing set
![Screenshot 2025-03-19 090807](https://github.com/user-attachments/assets/5d85cfa1-6cba-47eb-ad0e-3f7a8fe8f27d)

![image](https://github.com/user-attachments/assets/b4892b76-a40c-439f-8762-cf02ce40e23d)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
