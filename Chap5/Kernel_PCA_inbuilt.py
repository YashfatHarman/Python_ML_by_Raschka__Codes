'''
Scikit-learn comes with a kernel PCA class already in-built.

Let's use that on the half-moon dataset.
'''

from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np

X,y = make_moons(n_samples = 100, random_state = 123)

#plt.scatter(X[y==0, 0], X[y==0, 1], color = 'red', marker = '^', alpha = 0.5)
#plt.scatter(X[y==1, 0], X[y==1, 1], color = 'blue', marker = 'o', alpha = 0.5)

#plt.show()


scikit_kpca = KernelPCA(n_components = 2, kernel = "rbf", gamma = 15)
X_skernpca = scikit_kpca.fit_transform(X)    
    
plt.scatter(X_skernpca[y==0, 0], X_skernpca[y==0, 1], color = "red", marker = "^", alpha = 0.5)
plt.scatter(X_skernpca[y==1, 0], X_skernpca[y==1, 1], color = "blue", marker = "o", alpha = 0.5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

