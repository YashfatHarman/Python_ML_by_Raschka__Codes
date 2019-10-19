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
from sklearn.pipeline import Pipeline

from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC


pipe_svc = Pipeline([ ("scl", StandardScaler()), ("clf", SVC(random_state = 1))])

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [ {'clf__C': param_range, 'clf__kernel': ["linear"]}, {"clf__C": param_range, "clf__gamma" : param_range, "clf__kernel": ["rbf"] }]

gs = GridSearchCV(estimator = pipe_svc, param_grid = param_grid, scoring = "accuracy", cv = 10, n_jobs = -1)

gs = gs.fit(X_train, y_train)

print("best score: ", gs.best_score_)
print("best params: ", gs.best_params_)

#now test performance with the test dataset
clf = gs.best_estimator_
clf.fit(X_train, y_train)
print("Test accuracy: {:.3f}".format(clf.score(X_test, y_test)))

