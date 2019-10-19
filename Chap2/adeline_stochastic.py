import numpy as np
from numpy.random import seed

class AdelineSGD:
	'Adaptive Linear Neuron Classifier for stochastic gradient descent'
	'''
	eta: learning rate, between 0.0 and 1.0
	n_iter: no. of passes (epochs) over the training dataset
	
	shuffle: bool (default: True)
		Shuffles training data every epoch if set to True to prevent cycles 
	random_state: int (default None)
		Set random state for shuffling and initializing the weights
		
	'''
	
	def __init__(self, eta = 0.01, n_iter = 10, shuffle = True, random_state = None):
		self.eta = eta
		self.n_iter = n_iter
		self.w_initialized = False
		self.shuffle = shuffle
		
		if (random_state):
			seed(random_state)
		pass
		
	def _initialize_weights(self, m):
		'initialize weighs to zero'
		self.w_ = np.zeros(1 + m)
		self.w_initialized = True	
	
	def _shuffle(self,X,y):
		'Shuffles training data'
		r = np.random.permutation(len(y))
		return X[r],y[r]
	
	def fit(self, X, y):
		#basically this is where the model learns
		'''
		X: array-like; dimension: n_samples * n_features
		y: array-like; demension: n_samples 
		'''
		
		self._initialize_weights(X.shape[1])
		self.cost_ = []
		
		for i in range(self.n_iter):
			if self.shuffle:
				X,y = self._shuffle(X,y)
			cost = []
			for xi, target in zip(X,y):
				cost.append(self._update_weights(xi, target))
			avg_cost = sum(cost)/len(y)
			self.cost_.append(avg_cost)
		#print("cost_ len: ", len(self.cost_))
		return self
		
		pass
		
	def partial_fit(self,X,y):
		'Fit training data without reinitializing the weights'
		if not self.w_initialized:
			self._initialize_weights(X.shape[1])
		if y.ravel().shape[0] > 1:
			for xi,target in zip(X,y):
				self._update_weights(xi, target)
		else:
			self._update_weights(X,y)
		return self
					
		
	def _update_weights(self,xi, target):
		output = self.net_input(xi)
		#print("output.shape: ",output.shape())
		error = target - output
		#print("error.shape: ",error.shape())
		self.w_[1:] += self.eta * xi.dot(error)
		self.w_[0] += self.eta * error
		cost = 0.5 * error**2
		return cost 
		pass	
		
	def net_input(self,X):
		return np.dot(X, self.w_[1:]) + self.w_[0] 
				#X does not contain the traditional x0 = 1 in front. 
				#So have to deal the first one separately.
		pass
	
	def activation(self, X):
		return self.net_input(X)
		pass
	
	def predict(self, X):
		return np.where(self.activation(X) > 0.0, 1, -1)
		pass
		
		
	pass
