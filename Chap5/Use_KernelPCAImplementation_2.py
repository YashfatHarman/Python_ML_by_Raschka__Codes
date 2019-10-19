from KernelPCAImplementation import rbf_kernel_pca

'''
Let's use our implementaion of kernel PCA to separate concentric circles
'''

from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import numpy as np

X,y = make_circles(n_samples = 1000, random_state = 123, noise = 0.1, factor = 0.2)

plt.scatter(X[y==0, 0], X[y==0, 1], color = 'red', marker = '^', alpha = 0.5)
plt.scatter(X[y==1, 0], X[y==1, 1], color = 'blue', marker = 'o', alpha = 0.5)

plt.show()

'''
Clearly the cocentric circles are not linearly separable. So we would like to unfold them via kernel PCA so that the dataset can serve as a suitable input for linear classifier.

But first, let use see what the dataset looks like if we project it onto the principal components via standard PCA.
'''

from sklearn.decomposition import PCA
scikit_pca = PCA(n_components = 2)
X_spca = scikit_pca.fit_transform(X)

fig,ax = plt.subplots(nrows = 1, ncols = 2, figsize = (7,3))

ax[0].scatter(X_spca[y==0, 0], X_spca[y==0,1], color = "red", marker = "^", alpha = 0.5)
ax[0].scatter(X_spca[y==1, 0], X_spca[y==1,1], color = "blue", marker = "o", alpha = 0.5)

ax[1].scatter(X_spca[y==0, 0], np.zeros((500,1))+ 0.02, color = "red", marker = "^", alpha = 0.5)
ax[1].scatter(X_spca[y==1, 0], np.zeros((500,1))- 0.02, color = "blue", marker = "o", alpha = 0.5)

ax[0].set_xlabel("PC1")
ax[0].set_ylabel("PC2")
ax[1].set_ylim([-1,1])
ax[1].set_yticks([])
ax[1].set_xlabel("PC1")
plt.show()


'''
Now let's try out our kernel PCA function rbf_kernel_pca.
'''

from matplotlib.ticker import FormatStrFormatter
X_kpca = rbf_kernel_pca(X, gamma=15, n_components = 2)
    #gamma is a value we have to predict beforehand. Hopefully we'll see later how we can experiment to get a suitable value

fig,ax = plt.subplots(nrows = 1, ncols =2, figsize = (7,3))
ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0,1], color = "red", marker = "^", alpha = 0.5)
ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1,1], color = "blue", marker = "o", alpha = 0.5)

ax[1].scatter(X_kpca[y==0, 0], np.zeros((500,1))+ 0.02, color = "red", marker = "^", alpha = 0.5)
ax[1].scatter(X_kpca[y==1, 0], np.zeros((500,1))- 0.02, color = "blue", marker = "o", alpha = 0.5)

ax[0].set_xlabel("PC1")
ax[0].set_ylabel("PC2")
ax[1].set_ylim([-1,1])
ax[1].set_yticks([])
ax[1].set_xlabel("PC1")
ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
plt.show()
