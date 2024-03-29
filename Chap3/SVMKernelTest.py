from plotDecisionRegions import plot_decision_regions

import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC

np.random.seed(0)	#for reproducability

X_xor = np.random.randn(200,2)  #get random numbers for 200 rows and 2 columns
								#randn returns random floats from a range with mean 0 and variance 1

y_xor = np.logical_xor(X_xor[:,0] > 0, X_xor[:,1] > 0)	
								# this is neat. This will be False if both column 0 and 1 are of same sign.
								# True otherwise

y_xor = np.where(y_xor,1,-1)	#converting True/False values to 0 and 1.

plt.scatter(X_xor[y_xor==1, 0], X_xor[y_xor==1, 1], c = 'b', marker = 'x', label = '1')

plt.scatter(X_xor[y_xor==-1, 0], X_xor[y_xor==-1, 1], c = 'r', marker = 's', label = '-1')

plt.ylim(-3.0)

plt.legend()

plt.show()


svm = SVC(kernel = 'rbf', random_state = 0, gamma = 0.10, C = 10.0)

svm.fit(X_xor, y_xor)

plot_decision_regions(X_xor, y_xor, classifier = svm)

plt.legend(loc = 'upper left')

plt.show()
