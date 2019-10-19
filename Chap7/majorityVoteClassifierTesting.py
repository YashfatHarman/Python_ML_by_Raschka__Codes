from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

iris = datasets.load_iris()
X,y = iris.data[50:, [1,2]], iris.target[50:]

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 1)

# we'll now use three different classifiers: a logistic regression, a decision tree, a k-nearest neighbor
# we'll evaluate their individual performance via a 10-fold cross-validation on the training datased before we combine them into an ensemble clasifier 

from sklearn.cross_validation import cross_val_score 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

import numpy as np

clf1 = LogisticRegression(penalty = "l2", C = 0.001, random_state = 0)

clf2 = DecisionTreeClassifier(max_depth = 1, criterion = "entropy", random_state = 0)

clf3 = KNeighborsClassifier(n_neighbors = 1, p = 2, metric = "minkowski")

pipe1 = Pipeline([ ["sc", StandardScaler()],["clf", clf1] ])

pipe3 = Pipeline([ ["sc", StandardScaler()],["clf", clf3] ])

clf_labels = ["Logistic Regression", "Decision Tree", "KNN"]

print("10-fold cross validation: \n")

for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
    scores = cross_val_score(estimator = clf, X = X_train, y = y_train, cv = 10, scoring = "roc_auc")
    print("ROC AUC: {:.2f} +- {:.2f} [{}]".format(scores.mean(), scores.std(), label))
    
    
#now let's move on and combne the individual classifiers from majority rule voting in our MajorityVoteClassifier

from majorityVoteClassifier import MajorityVoteClassifier
mv_clf = MajorityVoteClassifier(classifiers = [pipe1, clf2, pipe3])

clf_labels += ["Majority Voting"]
all_clf = [pipe1, clf2, pipe3, mv_clf]

for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator = clf, X = X_train, y = y_train, cv = 10, scoring = "roc_auc" )
    print("Accuracy: {:.2f} +- {:.2f} [{}]".format(scores.mean(), scores.std(), label))
    

 


# now compute ROC curves from the test set to see if the modelgeneralizes well with unseen data

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import matplotlib.pyplot as plt

colors = ["black", "orange", "blue", "green"]
linestyles = [":", "--", "-.", "-"]

for clf,label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):
    #assuming the lable of the positive class is 1
    y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
    fpr,tpr,thresholds = roc_curve(y_true = y_test, y_score = y_pred)
    roc_auc = auc(x=fpr, y=tpr)
    
    plt.plot(fpr, tpr, color = clr, linestyle = ls, label = "{} (auc = {:.2f})".format(label, roc_auc))
    

plt.legend(loc = "lower right")
plt.plot([0,1], [0,1], linestyle = "--", color = "gray", linewidth = 2) 

plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])   
plt.grid()
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.show()


#now plot the decision regions.
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

from plotDecisionRegions import plot_decision_regions
from itertools import product

f,axarr = plt.subplots(nrows = 2, ncols = 2, sharex = "col", sharey = "row", figsize = (7,5))

for idx, clf, tt in zip(product([0,1],[0,1]), all_clf, clf_labels):
    clf.fit(X_train_std, y_train)
    plot_decision_regions(axarr[idx[0],idx[1]], X_train_std, y_train, classifier = clf, title = tt)

plt.text(-3.5, -4.5, s = "Sepal width [Standardized]", ha = "center", va = "center", fontsize = 12)
plt.text(-10.5, 4.5, s = "Petal width [Standardized]", ha = "center", va = "center", fontsize = 12, rotation = 90)

plt.show()


#get the parameters of the ensemble method
#print(mv_clf.get_params())

#now do a grid search to find the best parameters for the model.
#Let's work with inverse regularization parameter C of the logistic regression classifier and the decision tree depth

from sklearn.grid_search import GridSearchCV
params = {"decisiontreeclassifier__max_depth": [1,2], "pipeline-1__clf__C":[0.001, 0.1, 100.0]}
grid = GridSearchCV(estimator = mv_clf, param_grid = params, cv = 10, scoring = "roc_auc")
grid.fit(X_train, y_train)

for params, mean_score, scores in grid.grid_scores_:
    print("{:.3f} +- {:.2f} {}".format(mean_score, scores.std() /2, params))

print("Best parameters: ", grid.best_params_)

print("Accuracy: {:.2f}".format(grid.best_score_))
