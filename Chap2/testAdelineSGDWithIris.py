import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from perceptron import Perceptron
from adeline_stochastic import AdelineSGD

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

#standardization; feature-scaling
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

ada = AdelineSGD(n_iter = 15, eta = 0.01, random_state = 1)
ada.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada)

plt.title("Adeline - Stochastic Gradient Descent")
plt.xlabel("sepal length [std]")
plt.ylabel("petal length [std]")
plt.legend(loc = "upper left")
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker = "o")
plt.xlabel("epochs")
plt.ylabel("avg cost")
plt.show()


#code without feature scaling
'''
fig, ax = plt.subplots(nrows = 1,ncols = 2, figsize = (8,4))

ada1 = Adeline(n_iter = 10, eta = 0.01).fit(X,y)

ax[0].plot(range(1, len(ada1.cost_)+1), np.log10(ada1.cost_), marker = "o")

ax[0].set_xlabel("epochs")
ax[0].set_ylabel("log(sum-squared-errors)")
ax[0].set_title("Adeline - learning rate 0.01")

ada2 = Adeline(n_iter = 10, eta = 0.001).fit(X,y)

ax[1].plot(range(1, len(ada2.cost_)+1), np.log10(ada2.cost_), marker = "x")

ax[1].set_xlabel("epochs")
ax[1].set_ylabel("log(sum-squared-errors)")
ax[1].set_title("Adeline - learning rate 0.001")

plt.show()
'''


