'''
SBS => Sequential Backward Selection
SBS is used to remove unnecessary features from a dataset.
SBS is not implemented in scikit-learn, but it is easy to code from scratch.
'''

from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

class SBS():

	def __init__(self, estimator, k_features, scoring = accuracy_score, test_size = 0.25, random_state = 1):
		self.scoring = scoring
		self.estimator = clone(estimator)
		self.k_features = k_features
		self.test_size = test_size
		self.random_state = random_state
		pass
		
	def fit(self, X, y):
		
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = self.test_size, random_state = self.random_state)
		
		dim = X_train.shape[1]
		#print("dim:", dim)
		
		self.indices_ = tuple(range(dim))
		#print("indices_:", self.indices_)
		
		self.subsets_ = [self.indices_]
		#print("subsets_:", self.subsets_)
		
		score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
		#print("score:", score)
		
		self.scores_ = [score]
		#print("scores_", self.scores_)
		
		#print("getting into the loop:")
		while dim > self.k_features:
			#print("-----------------------------------")
			#print("dim:", dim)
			scores = []
			subsets = []
			
			for p in combinations(self.indices_, r=dim-1):
				score = self._calc_score(X_train, y_train, X_test, y_test, p)
				scores.append(score)
				subsets.append(p)
			
			#print("scores:", scores)
			#print("subsets:",subsets)
			
			best = np.argmax(scores)
			#print("best:",best)
			self.indices_ = subsets[best]
			#print("indices_:", self.indices_)
		
			self.subsets_.append(self.indices_)
			#print("subsets_:", self.subsets_)
		
			dim -= 1
					
			self.scores_.append(scores[best])
			#print("scores_", self.scores_)
		
		self.k_score_ = self.scores_[-1]		
		#print("k_score_:",self.k_score_)
		
		return self
		
		pass	
		
	def transform(self, X):
		return X[:, self.indices_]
		pass

	def _calc_score(self, X_train, y_train, X_test, y_test, indices):
		
		self.estimator.fit(X_train[:, indices], y_train)
		y_pred = self.estimator.predict(X_test[:, indices])
		score = self.scoring(y_test, y_pred)
		return score
		
		pass
			
	#end class
	
'now test our implementation with the wine data'
import pandas as pd
from sklearn.cross_validation import train_test_split

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
	

'ok, data is ready. Now do the actual feature selection.'
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

knn = KNeighborsClassifier(n_neighbors = 2)

sbs = SBS(knn, k_features = 1)

sbs.fit(X_train_std, y_train)


k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker = 'o')
plt.ylim([0.7, 1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()

k5 = list(sbs.subsets_[8])
print(df_wine.columns[1:][k5])

'Lets evaluate the performance of the KNN classifier on the original test set'
knn.fit(X_train_std, y_train)
print("training accuracy: ", knn.score(X_train_std, y_train))
print("test accuracy: ", knn.score(X_test_std, y_test))

'use the selected 5-feature subset and see how KNN performs'
knn.fit(X_train_std[:,k5], y_train)
print("training accuracy: ", knn.score(X_train_std[:,k5], y_train))
print("test accuracy: ", knn.score(X_test_std[:,k5], y_test))

