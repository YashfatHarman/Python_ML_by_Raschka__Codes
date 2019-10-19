import pandas as pd

df = pd.read_csv("housing.data", header = None, sep = "\s+")

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

X = df[['RM']].values 
y = df['MEDV'].values 


from sklearn.linear_model import LinearRegression

slr = LinearRegression()
slr.fit(X, y)

print("Slope: {:.3f}".format(slr.coef_[0]))
print("Intercept: {:.3f}".format(slr.intercept_))

#let's visualize how the linear regression line fits the training data
import matplotlib.pyplot as plt
def lin_regplot(X, y, model):
    plt.scatter(X, y, c = 'blue')
    plt.plot(X, model.predict(X), color = 'red')
    return None
    
lin_regplot(X,y,slr)
plt.xlabel("avg number of rooms [RM] (unstandradized)")
plt.ylabel("price in $1000\'s [MEDV] (unstandardized)")
plt.show()
