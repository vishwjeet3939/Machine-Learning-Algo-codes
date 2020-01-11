import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import datasets

dataset = pd.read_csv("Salary_Data.csv")

X = dataset.iloc[:, : -1].values
y = dataset.iloc[:, 1].values

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 1/3,random_state=0)

#fitting simple linear regression to the training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predcting the test set result
y_pred = regressor.predict(X_test)

#visualising the training set result
plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('salary vs experience(Training set')
plt.xlabel('Year of experience')
plt.ylabel('Salary')
plt.show()

#visualising the test set result
plt.scatter(X_test,y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('salary vs experience(Test set)')
plt.xlabel('Year of experience')
plt.ylabel('Salary')
plt.show()