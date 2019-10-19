import pandas as pd

df_wine = pd.read_csv("wine.data", header = None)

df_wine.columns = ["Class label", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols", "proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]

df_wine = df_wine[df_wine["Class label"] != 1]

# we only consider wine types 2 and 3, and we select two features only: Alcohol and Hue
y = df_wine["Class label"].values 
X = df_wine[["Alcohol","Hue"]].values

#now encode the dataset into binary format and split the dataset into 60% training and 40% test dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state = 1)

# we'll use an unpruned decision tree as the base classifier.
# and create an ensemble of 500 decision trees fitted on different bootstrap samples of the training dataset.

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

tree = DecisionTreeClassifier(criterion = "entropy", max_depth = None)

bag = BaggingClassifier(base_estimator = tree, n_estimators = 500, max_samples = 1.0, max_features = 1.0, bootstrap = True, bootstrap_features = False, n_jobs = 1, random_state = 1)

#compare the performance of the bagging classifier with a single unpruned decision tree
from sklearn.metrics import accuracy_score

tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print("Decision tree train/test accuracies: {:.3f} / {:.3f}".format(tree_train, tree_test))

bag = bag.fit(X_train, y_train)
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)
bag_train = accuracy_score(y_train, y_train_pred)
bag_test = accuracy_score(y_test, y_test_pred)
print("Bagging train/test accuracies: {:.3f} / {:.3f}".format(bag_train, bag_test))

#plot decision bounderies
from plotDecisionRegions import plot_decision_regions
import matplotlib.pyplot as plt

f, axarr = plt.subplots(nrows = 1, ncols = 2, sharex = "col", sharey = "row", figsize = (8,3))

for idx, clf, tt in zip([0,1], [tree, bag], ["Decision Tree","Bagging"]):
    clf.fit(X_train, y_train)
    plot_decision_regions(axarr[idx], X_train, y_train, classifier = clf, title = tt)

plt.text(-10.2, -1.2, s = "Hue", ha = "center", va = "center", fontsize = 12)
plt.text(-10.5, 4.5, s = "Alcohol", ha = "center", va = "center", fontsize = 12, rotation = 90)

plt.show()

