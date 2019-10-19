import pandas as pd
import matplotlib.pyplot as plt
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

from sklearn.metrics import roc_curve, auc
from scipy import interp

from sklearn.cross_validation import StratifiedKFold

X_train2 = X_train[:,[4,14]]    #just selcting two features for prediction
cv = StratifiedKFold(y_train, n_folds = 3, random_state = 1)

fig = plt.figure(figsize = (7,5))
mean_tpr = 0.0
mean_fpr = np.linspace(0,1,100)
all_tpr = []

for i,(train, test) in enumerate(cv):
    probas = pipe_lr.fit(X_train2[train], y_train[train]).predict_proba(X_train2[test])
    fpr, tpr, threasholds = roc_curve(y_train[test], probas[:,1], pos_label = 1)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw = 1, label = "ROC fold {:d} (area = {:.2f})".format(i+1, roc_auc))
    
plt.plot([0,1], [0,1], linestyle = "--", color = (0.6, 0.6, 0.6), label = "random guessing")

mean_tpr /= len(cv)
mean_tpr[-1] = 1.0

mean_auc = auc(mean_fpr, mean_tpr)

plt.plot(mean_fpr, mean_tpr, "k--", label = "mean ROC (area = {:.2f})".format(mean_auc), lw = 2)

plt.plot([0,0,1], [0,1,1], lw = 2, linestyle = ":", color = "black", label = "perfect performance")

plt.xlim([-0.5, 1.05])
plt.ylim([-0.5, 1.05])
plt.xlabel("flase positive rate")
plt.ylabel("true positive rate")
plt.title("Receiver Operator Characteristic")
plt.legend(loc= "lower right")
plt.show()

#getting the ROC AUC score directly
from sklearn.svm import SVC
pipe_svc = Pipeline([ ("scl", StandardScaler()), ("clf", SVC(random_state = 1))])
pipe_svc = pipe_svc.fit(X_train2, y_train)
y_pred2 = pipe_svc.predict(X_test[:, [4,14]])

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

print("ROC AUC: {:.3f}".format(roc_auc_score(y_true = y_test, y_score = y_pred2) ) )

print("Accuracy: {:.3f}".format(accuracy_score(y_true = y_test, y_pred = y_pred2)))

