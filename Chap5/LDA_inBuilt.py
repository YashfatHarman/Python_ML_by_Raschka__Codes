import pandas as pd
import numpy as np
from plotDecisionRegions import plot_decision_regions 
import matplotlib.pyplot as plt

df_wine = pd.read_csv("wine.data", header = None)
print(df_wine.shape)

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

X,y = df_wine.iloc[:, 1:].values, df_wine.loc[:,0].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

#from sklearn.lda import LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components = 2)
X_train_lda = lda.fit_transform(X_train_std, y_train)

#lets see how the logistic regression classifier handles the lower-dimensional training dataset after the LDA transformation

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr = lr.fit(X_train_lda, y_train)
plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel("LD 1")
plt.ylabel("LD 2")
plt.legend(loc = "lower left")
plt.show()

X_test_lda = lda.transform(X_test_std)
plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel("LD 1")
plt.ylabel("LD 2")
plt.legend(loc = "lower left")
plt.show()
