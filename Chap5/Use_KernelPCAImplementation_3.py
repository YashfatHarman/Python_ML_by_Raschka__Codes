from KernelPCAImplementation2 import rbf_kernel_pca

'''
Let's use our implementaion of kernel PCA to separate half-moon shapes

This time we try with test data as well.
'''


from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np

X,y = make_moons(n_samples = 100, random_state = 123)

plt.scatter(X[y==0, 0], X[y==0, 1], color = 'red', marker = '^', alpha = 0.5)
plt.scatter(X[y==1, 0], X[y==1, 1], color = 'blue', marker = 'o', alpha = 0.5)

plt.show()

'''
Clearly the half-moons are not linearly separable. So we would like to unfoldthem via kernel PCA so that the dataset can serve as a suitable input for linear classifier.
'''

from matplotlib.ticker import FormatStrFormatter
alphas, lambdas = rbf_kernel_pca(X, gamma=15, n_components = 1)
    #gamma is a value we have to predict beforehand. Hopefully we'll see later hw we can experiment to get a suitable value
    
    
'''
To make sure that we implement the code for projecting new samples, let's assume that the 26th point from the half-moon dataset is a new data pont x', and our task is the project it onto this new subspace.
'''    

x_new = X[25]
x_proj = alphas[25] #original projection

'''
this function can project any new data point. 
So we will be able to verify that the projection will work for new data by checking if X[25] projects at the same place as in x_proj above.  
'''
def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new-row)**2) for row in X])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas/lambdas)    #we are normalizing eigenvectors by corresponding eigenvalues
    pass
    
x_reproj = project_x(x_new, X, gamma = 15, alphas=alphas, lambdas = lambdas)


plt.scatter(alphas[y==0, 0], np.zeros((50)), color = "red", marker = "^", alpha = 0.5)
plt.scatter(alphas[y==1, 0], np.zeros((50)), color = "blue", marker = "o", alpha = 0.5)
plt.scatter(x_proj, 0, color = "black", label = "original projection of point X[25]", marker = "^", s = 100)
plt.scatter(x_reproj, 0, color = "green", label = "remapped point X[25]", marker = "x", s = 500)
plt.legend(scatterpoints = 1)
plt.show()

