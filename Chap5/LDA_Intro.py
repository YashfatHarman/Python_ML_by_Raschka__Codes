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

import numpy as np

np.set_printoptions(precision=4)
mean_vecs = []

#take mean of each class; so one mean for values where y = 0, one for where y = 1, etc.
for label in range(1,4):
    mean_vecs.append( np.mean(X_train_std[y_train == label], axis = 0) )
    print("MV: {} : {}".format(label, mean_vecs[label-1]))
    

#calculate the within class scatter matrix
#scaled withing class scatter matrix is the same as the covariance matrix
d = 13 #number of features
S_W = np.zeros((d,d))

for label, mv in zip( range(1,4), mean_vecs):
    class_scatter = np.cov( X_train_std[y_train == label].T )
    S_W += class_scatter
    

#now compute the between class scatter matrix
mean_overall = np.mean(X_train_std, axis = 0)
d = 13 #no of features
S_B = np.zeros( (d,d) )
for i, mean_vec in enumerate(mean_vecs):
    n = X[y == i+1, :].shape[0] #counts no of rows of a particular class
    mean_vec = mean_vec.reshape(d,1) #converting a row matrix to a column matrix
    mean_overall = mean_overall.reshape(d,1)
    S_B += n * (mean_vec - mean_overall).dot((mean_vec-mean_overall).T)

#both S_W and S_B are 13*13 in this case

eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

#sort the eigenvalues by descending order
eigen_pairs = [ (np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals)) ] #only considering absolute values of eigenvalues, not interested in their directions
eigen_pairs = sorted(eigen_pairs, key = lambda k: k[0], reverse = True) #sort based on eigenvalues

print("eigenvalues in decreasing order:")
for eigen_pair in eigen_pairs:
    print(eigen_pair[0])
    
    
import matplotlib.pyplot as plt

tot = sum(eigen_vals.real)
discr = [(i/tot) for i in sorted(eigen_vals.real, reverse = True)]
cum_discr = np.cumsum(discr)
plt.bar(range(1,14), discr, alpha = 0.5, align = "center", label = "individual discriminablity")
plt.step(range(1,14), cum_discr, where = "mid", label = "cumulative discriminablity")
plt.ylabel("discriminablity ratio")
plt.xlabel("linear discriminants")
plt.ylim( [-0.1, 1.1] )
plt.legend(loc = "best")
plt.show()

#now create the transformation matrix
W = np.hstack((eigen_pairs[0][1][:, np.newaxis].real, eigen_pairs[1][1][:, np.newaxis].real))
print("W:", W)

#transform the training set
X_train_lda = X_train_std.dot(W)

colors = ["r","g", "b"]
markers = ["s","x","o"]
for l,c,m in zip(np.unique(y_train),colors,markers):
    plt.scatter(X_train_lda[y_train==l, 0], X_train_lda[y_train==l, 1], c=c, label=l, marker=m )
plt.xlabel("LD 1")
plt.ylabel("LD 2")
plt.legend(loc = "upper right")
plt.show()

