'Implementation of RBF kernel PCA in python'
'Modified to be able to work with test data as well.'

from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np

def rbf_kernel_pca(X, gamma, n_components):
    
    """
    RBF Kernel PCA implementation
    
    Parameters:
    -----------
    X: {Numpy ndarray}, shape = [n_samples, n_features]
    
    gamma: float
        Tuning parameter of the RBF kernel
    
    n_components: int
        Number of principal components to return
        
    Returns:
    ----------
    X_pc: {Numpy ndarray}, shape = [n_samples, k_features]
        Projected dataset
        
    lambdas: list
        Eigenvalues
    """
    
    #calculate pairwise squared Eucledean distances in the M x N dimension dataset
    sq_dists = pdist(X, "sqeuclidean")   #gets pairwise distance between observations in a n-dimensional space
    
    #convert pairwise distances into a square matrix
    mat_sq_dists = squareform(sq_dists) 
        #squareform converts a vector-form distance vector to a square-form distance matrix, and vice-versa
        #here it is needed because pdist returned an array of shape combination(M,2) x 1. Need to convert it to square matrix.
        
    #compute the symmetric kernel matrix
    K = exp(-gamma * mat_sq_dists)
    
    #center the kernel matrix
    N = K.shape[0]
    one_n = np.ones( (N,N) ) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    
    #obtaining eigenpairs from the centered kernel matrix
    #scipy.linalg.eigh returns them in sorted order
    eigvals, eigvecs = eigh(K)
    
    #collect the top k eigenvectors (proected samples)
    alphas = np.column_stack( (eigvecs[:,-i] for i in range(1, n_components + 1)) )
    
    #cllect the corresponding eigenvalues
    lambdas = [eigvals[-i] for i in range(1, n_components + 1)]
    
    return alphas, lambdas
    
    
    pass
