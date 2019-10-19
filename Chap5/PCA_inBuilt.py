from plotDecisionRegions import plot_decision_regions 

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

import pandas as pd

df_wine = pd.read_csv("wine.data", header = None)
print(df_wine.shape)

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

X,y = df_wine.iloc[:, 1:].values, df_wine.loc[:,0].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


pca = PCA(n_components = 2)

lr = LogisticRegression()

X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

lr.fit(X_train_pca, y_train)

import matplotlib.pyplot as plt
plot_decision_regions(X_train_pca, y_train, classifier = lr)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(loc="lower left")
plt.show()

#let's check with test data to see if the logistic regression worked
plot_decision_regions(X_test_pca, y_test, classifier = lr)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(loc="lower left")
plt.show()

