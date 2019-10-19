import pandas as pd
import numpy as np

df = pd.read_csv("housing.data", header = None, sep = "\s+")

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
 
from sklearn.cross_validation import train_test_split

X = df.iloc[:, :-1].values
y = df['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0 )

from sklearn.linear_model import LinearRegression

slr = LinearRegression()
slr.fit(X_train, y_train)

y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

#we can't plot the training data directly as they are multidimesional.
#So let's plot the error residuals against the predictions.

import matplotlib.pyplot as plt

plt.scatter(y_train_pred, y_train_pred - y_train, c ="blue", marker = "o", label = "Training data")
plt.scatter(y_test_pred, y_test_pred - y_test, c ="lightgreen", marker = "s", label = "Test data")

plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y=0, xmin = -10, xmax = 50, lw = 2, color = "red")
plt.xlim([-10, 50])
plt.show()

from sklearn.metrics import mean_squared_error
print("MSE train: {:.3f}, test: {:.3f} %".format(mean_squared_error(y_train,y_train_pred), mean_squared_error(y_test,y_test_pred)))

from sklearn.metrics import r2_score
print("R^2 train: {:.3f}    test: {:.3f}".format(r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))
