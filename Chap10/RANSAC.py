import pandas as pd
import numpy as np

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

from sklearn.linear_model import RANSACRegressor

ransac = RANSACRegressor(LinearRegression(), max_trials = 100, min_samples = 50, residual_metric = lambda x: np.sum(np.abs(x), axis = 1), residual_threshold = 5.0, random_state = 0 )
    # max number of iterations is 100
    # minimum number of randomly chosen samples is 50
    # residual_metric is a callable lambda function that just measures the vertial distances between the fitted line and the sample points
    # samples with their vertical distances to the fitted line within 5.0 will be included in the inliner set
     
ransac.fit(X,y) 

#let's see which samples were inliers and which were outliers
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])

plt.scatter(X[inlier_mask], y[inlier_mask], c = 'blue', marker = 'o', label = 'Inliers')

plt.scatter(X[outlier_mask], y[outlier_mask], c = 'lightgreen', marker = 's', label = 'Outliers')

plt.plot(line_X, line_y_ransac, color = 'red')

plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.legend(loc = 'upper left')
plt.show()

print("Slope: {:.3f}".format(ransac.estimator_.coef_[0]))
print("Intercept: {:.3f}".format(ransac.estimator_.intercept_))

 


