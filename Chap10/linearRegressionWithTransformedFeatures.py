import pandas as pd
import numpy as np

df = pd.read_csv("housing.data", header = None, sep = "\s+")

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

X = df[['LSTAT']].values 
y = df['MEDV'].values 

#transform features
X_log = np.log(X)
y_sqrt = np.sqrt(y) 

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

regr = LinearRegression()

#fit features
X_fit = np.arange(X_log.min() - 1, X_log.max() + 1, 1)[:, np.newaxis]
regr = regr.fit(X_log, y_sqrt)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y_sqrt, regr.predict(X_log))

#plot results
import matplotlib.pyplot as plt
plt.scatter(X_log, y_sqrt, label = "training points", color = "lightgray")
plt.plot(X_fit, y_lin_fit, label = "linear(d=1), $R^2 = {:.2f}$".format(linear_r2), color = "blue", lw = 2)
plt.xlabel("log (% lower status of the population [LSTAT])")
plt.ylabel("$sqrt{Price \; in \; \$1000\'s [MEDV]}$")
plt.legend(loc ="lower left")
plt.show()
