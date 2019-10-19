import pandas as pd

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

#we want to:
#    1. standardize the features
#    2. compress data from 30 to 2 dimension space using PCA
#    3. Apply Logistic Regression

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe_lr = Pipeline([("scl", StandardScaler()), ("pca", PCA(n_components = 2)), ("clf", LogisticRegression(random_state=1)) ])

pipe_lr.fit(X_train, y_train)

print("Test Accuracy: {:.3f}".format(pipe_lr.score(X_test, y_test)))

#now do cross-validation
#we'll try stratified cross-validation, where class proportions are preserved in each fold.
import numpy as np
from sklearn.cross_validation import StratifiedKFold
kfold = StratifiedKFold(y=y_train, n_folds = 10, random_state = 1)

scores = []
for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: {}, Class dist: {}, Accuracy: {:.3f}%'.format(k+1, np.bincount(y_train[train]), score))
    
print("CV accuracy: {:.3f} +- {:.3f}".format(np.mean(scores), np.std(scores)))

#let's use the in-built k-fold cross-validation scorer
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(estimator = pipe_lr, X=X_train, y=y_train, cv = 10, n_jobs = 1)
print("CV accuray score: ", scores)
print("CV accuracy: {:.3f} +- {:.3f}".format(np.mean(scores), np.std(scores)))


