## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
step 1.start

step 2. Importing necessary liberaries

step 3. Data preprocessing

step 4. Spliting data int training and testing data

step 5. Performing SGD-Regressor

step 6. Calculating error

step 7. end

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Vijayaraj V
RegisterNumber: 212222230174
*/
```
```
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data=fetch_california_housing()
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
print(df.head())
```
# Output:
![ML4 1](https://github.com/user-attachments/assets/7c51ca8c-c5ec-4e5d-a3c5-dd9c6fba6897)
```
print(df.tail())
```
# Output:
![ML4 2](https://github.com/user-attachments/assets/ea6118ef-a915-45f9-b2e2-067eee05d700)
```
X = df.drop(columns=['AveOccup','target'])
Y = df['target']
X.shape
Y.shape
X.info()
```
# Output:
![ML4 3](https://github.com/user-attachments/assets/fa7150f4-cd78-4b1b-987a-6df252c8c136)

```
X=data.data[:,:3]
Y=np.column_stack((data.target,data.data[:,6]))
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
x.head()
```
# Output:
![ML4 4](https://github.com/user-attachments/assets/56f59346-a50f-4f8c-bc24-dea4059c8a03)

```
scaler_X=StandardScaler()
scaler_Y=StandardScaler()
X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.transform(Y_test)

sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train,Y_train)
Y_pred=multi_output_sgd.predict(X_test)
Y_pred=scaler_Y.inverse_transform(Y_pred)
Y_test=scaler_Y.inverse_transform(Y_test)

mse=mean_squared_error(Y_test,Y_pred)
print("Mean Sqaured Error:",mse)
print("\nPredictions:\n",Y_pred[:5])
```
## Output:
![ML4 5](https://github.com/user-attachments/assets/26d1403a-22dc-4285-8138-666fbd6d5c7a)

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
