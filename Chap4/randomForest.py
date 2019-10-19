'Again, prepare the wine data'
import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np

df_wine = pd.read_csv("wine.data", header = None)

print(df_wine.shape)

df_wine.columns = ["Class label", "Alcohol", "Malic Acid", "Ash", "Alcalinity of Ash", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]

#print("Class labels",np.unique(df_wine["Class label"]))

#print(df_wine.head())

X,y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#now do feature scaling

#standardization
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
#print(X_train_std[:5,:])	


#now use Random Forest
from sklearn.ensemble import RandomForestClassifier

feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators = 10000, random_state = 0, n_jobs = -1) #what is n_jobs?

forest.fit(X_train, y_train)

importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("{:2d}) {:30s} {:10.4f}".format(f+1, feat_labels[f], importances[indices[f]]))

import matplotlib.pyplot as plt
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], color="lightblue", align = "center")
plt.xticks(range(X_train.shape[1]), feat_labels, rotation = 90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()
