import pandas as pd
from sklearn.cross_validation import train_test_split

df = pd.read_csv("housing.data", header = None, sep = "\s+")

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

X = df.iloc[:,:-1].values
y = df['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 1)

from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators = 1000, criterion = 'mse', random_state = 1, n_jobs = -1)

forest.fit(X_train, y_train)

y_train_pred = forest.predict(X_train)

y_test_pred = forest.predict(X_test)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

print("MSE train: {:.3f}, test: {:.3f}".format(mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))

print("R^2 train: {:.3f}, test: {:.3f}".format(r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))

#let's take a look at the residuals of the prediction
import matplotlib.pyplot as plt

plt.scatter(y_train_pred, y_train_pred - y_train, c = "black", marker = "o", s = 35, alpha = 0.5, label = "Training data")

plt.scatter(y_test_pred, y_test_pred - y_test, c = "lightgreen", marker = "s", s = 35, alpha = 0.7, label = "Test data")

plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = -10, xmax = 50, lw =2, color = "red")
plt.xlim([-10, 50])
plt.show()

