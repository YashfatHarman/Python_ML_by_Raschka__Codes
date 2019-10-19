import pandas as pd
from sklearn.cross_validation import train_test_split

df = pd.read_csv("housing.data", header = None, sep = "\s+")

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

X = df[['LSTAT']].values
y = df['MEDV'].values
#X = df.iloc[:,:-1].values
#y = df['MEDV'].values
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 1)


from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth = 3)

tree.fit(X, y)
#tree.fit(X_train, y_train)
#y_train_pred = tree.predict(X_train)
#y_test_pred = tree.predict(X_test)

#from sklearn.metrics import r2_score
#from sklearn.metrics import mean_squared_error

#print("MSE train: {:.3f}, test: {:.3f}".format(mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))

#print("R^2 train: {:.3f}, test: {:.3f}".format(r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))



sort_idx = X.flatten().argsort()

import matplotlib.pyplot as plt
def lin_regplot(X, y, model):
    plt.scatter(X, y, c = 'blue')
    plt.plot(X, model.predict(X), color = 'red')
    return None

lin_regplot(X[sort_idx], y[sort_idx], tree)

#plt.xlabel("% lower status of the population [LSTAT]")
#plt.ylabel('Price in $1000\'s [MEDV]')
#plt.show()

