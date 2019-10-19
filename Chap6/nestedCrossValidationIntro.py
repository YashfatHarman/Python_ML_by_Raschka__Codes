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
from sklearn.cross_validation import cross_val_score
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

scores = cross_val_score(gs, X, y, scoring = "accuracy", cv = 5)

print("CV accuracy of SVM model: {:.3f} +- {:.3f}".format(np.mean(scores), np.std(scores)))


#use the same nested cross validation approach to compare the performance of the svm to that of a decision tree with the same dataset

from sklearn.tree import DecisionTreeClassifier

gs = GridSearchCV(estimator = DecisionTreeClassifier(random_state = 0), param_grid = [{"max_depth":[1,2,3,4,5,6,7, None]}], scoring = "accuracy", cv = 5)
scores = cross_val_score(gs, X_train, y_train, scoring = "accuracy", cv = 5)
print("CV accuracy of decision tree model: {:.3f} +- {:.3f}".format(np.mean(scores), np.std(scores)))


