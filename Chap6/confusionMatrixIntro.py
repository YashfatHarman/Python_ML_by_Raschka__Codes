import pandas as pd
import matplotlib.pyplot as plt

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
from sklearn.svm import SVC
pipe_svc = Pipeline([ ("scl", StandardScaler()), ("clf", SVC(random_state = 1))])

from sklearn.metrics import confusion_matrix
pipe_svc.fit(X_train, y_train)

y_pred = pipe_svc.predict(X_test)

confmat = confusion_matrix(y_true = y_test, y_pred = y_pred)
print(confmat)

'''
format of confusion matrix:

            Predited class
               P    N
------------------------    
Actual  P   | TP    FN
Class   N   | FP    TN 

'''

#let's plot the confusion matrix
fig, ax = plt.subplots(figsize = (2.5, 2.5))
ax.matshow(confmat, cmap = plt.cm.Blues, alpha = 0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x = j, y = i, s = confmat[i, j], va = 'center', ha = 'center')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()

#let's check precision and recall
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
print("Precision: {:.3f}".format(precision_score(y_true = y_test, y_pred = y_pred)))
print("Recall: {:.3f}".format(recall_score(y_true = y_test, y_pred = y_pred)))
print("F1: {:.3f}".format(f1_score(y_true = y_test, y_pred = y_pred)))


