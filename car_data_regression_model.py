#aim: To predict the price of the car, using various features provided in the dataset
#Car Dataset: https://www.kaggle.com/nehalbirla/vehicle-dataset-from-cardekho?select=car+data.csv


#libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

#loading the data
df = pd.read_csv('cardata.csv')
#df = pd.read_csv('Samples/cardata.csv')
df.head()

#information about data
#year: Year of the car when it was bought
#selling price: Price at which the car is being sold
#present price: This is the current ex-showroom price of the car.
#km driven: Number of Kilometres the car is driven
#fuel: car is driven text_format fuel sort Fuel type of car (petrol / diesel / CNG / LPG / electric)
#seller type: Tells if a Seller is Individual or a Dealer
#transmission: Gear transmission of the car (Automatic/Manual)
#owner: Number of previous owners of the car.


#data preprocessing
df.info()
df.isnull().sum()


#encoding
binary_cols = [col for col in df.columns if df[col].dtype in ["O"] and col != "Car_Name"]

df = pd.get_dummies(df, columns=binary_cols, drop_first=False)

df.head()
df.info()

X=df.drop(["Car_Name","Selling_Price"], axis=1)
y=df["Selling_Price"]

#training set and test set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

#First Model: Linear Regression
lr= LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)

# MSE
mean_squared_error(y_test, y_pred)

# RMSE
np.sqrt(mean_squared_error(y_test, y_pred))

# MAE
mean_absolute_error(y_test, y_pred)

# R-KARE
r2_score(y_test, y_pred)


#Second Model: Polynomial Regression 

#Degree=2, Polynomial Regression
poly = PolynomialFeatures(degree = 2)
x_poly = poly.fit_transform(x_train)
lr.fit(x_poly,y_train)

y_pred2=lr.predict(x_poly)

# MSE
mean_squared_error(y_train, y_pred2)

# RMSE
np.sqrt(mean_squared_error(y_train, y_pred2))

# MAE
mean_absolute_error(y_train, y_pred2)

# R-KARE
r2_score(y_train, y_pred2)


# Degree=3,Polynomial Regression
poly3 = PolynomialFeatures(degree = 3)
x_poly3 = poly3.fit_transform(x_train)
lr.fit(x_poly3,y_train)

y_pred3=lr.predict(x_poly3)

# MSE
mean_squared_error(y_train, y_pred3)

# RMSE
np.sqrt(mean_squared_error(y_train, y_pred3))

# MAE
mean_absolute_error(y_train, y_pred3)

# R-KARE
r2_score(y_train, y_pred3)


#Degree=4, Polynomial Regression 
poly4 = PolynomialFeatures(degree = 4)
x_poly4 = poly4.fit_transform(x_train)
lr.fit(x_poly4,y_train)

y_pred4=lr.predict(x_poly4)

# MSE
mean_squared_error(y_train, y_pred4)

# RMSE
np.sqrt(mean_squared_error(y_train, y_pred4))

# MAE
mean_absolute_error(y_train, y_pred4)

# R-KARE
r2_score(y_train, y_pred4)


#Third Model: Decision Tree Regression
dt = DecisionTreeRegressor(random_state=0)
dt.fit(x_train,y_train)

y_pred5=dt.predict(x_test)

# MSE
mean_squared_error(y_test, y_pred5)

# RMSE
np.sqrt(mean_squared_error(y_test, y_pred5))

# MAE
mean_absolute_error(y_test, y_pred5)

# R-KARE
r2_score(y_test, y_pred5)


#Fourth Model: Support Vector Regression (SVR)

#Scaling Data
std=StandardScaler()
x_train=std.fit_transform(x_train)
x_test=std.fit_transform(x_test)

y_train=np.array(y_train)       #yapmak doğrumu
y_train=y_train.reshape(-1,1)
y_train=std.fit_transform(y_train)

y_test=np.array(y_test)
y_test=y_test.reshape(-1,1)
y_test=std.fit_transform(y_test)


svr=SVR(kernel='rbf')
svr.fit(x_train,y_train)
         
y_pred6=svr.predict(x_test)

# MSE
mean_squared_error(y_test, y_pred6)

# RMSE
np.sqrt(mean_squared_error(y_test, y_pred6))

# MAE
mean_absolute_error(y_test, y_pred6)

# R-KARE
r2_score(y_test, y_pred6)


#predict 
b=np.array(X[1:2])
b

a=np.array([[2014,10.79,15000,0,1,0,0,1,0,0,0]])
a

#support vector
y_pred7=svr.predict(b)
#support vector
y_pred7

y_pred8=svr.predict(a)
y_pre8a

#decision tree 
y_pred9=dt.predict(b)
y_pred9

y_pred10=dt.predict(a)
y_pred10

