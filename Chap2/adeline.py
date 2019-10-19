import numpy as np

class Adeline:
	'Adaptive Linear Neuron Classifier'
	'''
	eta: learning rate, between 0.0 and 1.0
	n_iter: no. of passes (epochs) over the training dataset
	'''
	
	def __init__(self, eta = 0.01, n_iter = 50):
		self.eta = eta
		self.n_iter = n_iter
		pass
	
	def fit(self, X, y):
		#basically this is where the model learns
		'''
		X: array-like; dimension: n_samples * n_features
		y: array-like; demension: n_samples 
		'''
		
		self.w_ = np.zeros(1 + X.shape[1])	#shape[1] is the second dimension of X.
											#means n_features
		self.cost_ = []
		
		for i in range(self.n_iter):
			output = self.net_input(X)
			errors = y - output
			self.w_[1:] += self.eta * X.T.dot(errors)
			self.w_[0] += self.eta * errors.sum()
			cost = (errors**2).sum() / 2.0
			self.cost_.append(cost)
		return self
		
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
