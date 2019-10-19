from linearRegression import LinearRegressioGD

import pandas as pd

df = pd.read_csv("housing.data", header = None, sep = "\s+")

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']


X = df[['RM']].values 
y = df['MEDV'].values 

print(X.shape)
print(y.shape)

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
sc_y = StandardScaler()

X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y)

lr = LinearRegressioGD()
lr.fit(X_std, y_std)

#plot the cost against number of epochs to see if the has converged
import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set(style = 'whitegrid', context = 'notebook')

plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.show()

#let's visualize how the linear regression line fits the training data
def lin_regplot(X, y, model):
    plt.scatter(X, y, c = 'blue')
    plt.plot(X, model.predict(X), color = 'red')
    return None
    
lin_regplot(X_std, y_std, lr)
plt.xlabel("avg number of roms [RM] (standradized)")
plt.ylabel("price in $1000\'s [MEDV] (standardized)")
plt.show()

#report the predicted price outcome in its original scale
num_rooms_std = sc_x.transform([5.0])
price_std = lr.predict(num_rooms_std)
print(price_std)
print(sc_y.inverse_transform(price_std))
print("Price in $1000's: {:.3f}".format(float(sc_y.inverse_transform(price_std))))

print("Slope: {:.3f}".format(lr.w_[1]))
print("Intercept: {:.3f}".format(lr.w_[0]))
