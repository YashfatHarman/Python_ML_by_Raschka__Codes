import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from perceptron import Perceptron

def plot_decision_regions(X, y, classifier, resolution=0.02):
	#setup marker generator and color map
	markers = ('s','x','o','^','v')
	colors = ("red", "blue", "lightgreen", "blue", "cyan")
	cmap = ListedColormap(colors[:len(np.unique(y))])
	
	#plot the decision surface
	x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1 
	x2_min, x2_max = X[:,1].min() - 1, X[:,1].max() + 1 
	
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution) )
	
	Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
	Z = Z.reshape(xx1.shape)
	
	plt.contourf(xx1,xx2,Z,alpha = 0.4,cmap = cmap)
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())
	
	#plot class samples
	for idx,cl in enumerate(np.unique(y)):
		plt.scatter(x = X[y==cl, 0], y = X[y==cl, 1], alpha= 0.8, c = cmap(idx), marker = markers[idx], label= cl)
	pass

df = pd.read_csv("iris.data", header=None)

y = df.iloc[0:100, 4].values

y = np.where(y == "Iris-setosa", -1, 1) #setosas will be tagged as -1

X = df.iloc[0:100, [0,2]].values

plt.scatter(X[:50, 0], X[:50, 1], color = "red", marker = "o", label = "setosa")
plt.scatter(X[50:100, 0], X[50:100, 1], color = "blue", marker = "x", label = "versicolor")

plt.xlabel("petal length")
plt.ylabel("sepal length")

plt.legend(loc = "upper left")

plt.show()

ppn = Perceptron(eta = 0.1, n_iter = 10)
ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker = 'o')
plt.xlabel("epoch")
plt.ylabel("no of misclassifications")
plt.show()

plot_decision_regions(X, y, classifier = ppn)
plt.xlabel("sepal len(cm)")
plt.ylabel("petal len(cm)")
plt.legend(loc = "upper left")
plt.show()
