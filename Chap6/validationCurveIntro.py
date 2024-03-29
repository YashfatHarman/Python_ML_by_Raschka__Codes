import pandas as pd
import numpy as np

df = pd.read_csv("wdbc.data", header = None)

print(df.shape)
    # shape is 569 * 32
    # column 0 for ID
    # column 1 for diagnosis (B = benign, M = Malignant)
    # the next 30 columns contain features.
    
from sklearn.preprocessing import LabelEncoder

X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y) #now malignant tumors are 1, benigns are 0

#divide the dataset into training-testing part (80-20 split)
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


pipe_lr = Pipeline([ ("scl", StandardScaler()), ("clf", LogisticRegression(penalty = "l2", random_state = 0) )])

import matplotlib.pyplot as plt
from sklearn.learning_curve import validation_curve

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

train_scores, test_scores = validation_curve(estimator = pipe_lr, X = X_train, y = y_train, param_name = "clf__C", param_range = param_range, cv = 10)
    #we want to evaluate the inverse regularization parameter C of LogisticRegression classifier. So it's written as clf__C to access the clf object in the pipleline.

train_mean = np.mean(train_scores, axis = 1)
train_std = np.std(train_scores, axis = 1)

test_mean = np.mean(test_scores, axis = 1)
test_std = np.std(test_scores, axis = 1)

plt.plot(param_range, train_mean, color = "blue", marker = "o", markersize = 5, label = "training accuracy")

plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha = 0.15, color = "blue")

plt.plot(param_range, test_mean, color = "green", marker = "s", markersize = 5, label = "validation accuracy")

plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha = 0.15, color = "green")

plt.grid()
plt.xscale("log")
plt.xlabel("Parameter C")
plt.ylabel("Accuracy")
plt.legend(loc = "lower right")
plt.ylim([0.8, 1.0])
plt.show()
