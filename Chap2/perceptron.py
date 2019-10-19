import numpy as np

class Perceptron:
	'''
	eta: learning rate, between 0.0 and 1.0
	n_iter: no. of passes (epochs) over the training dataset
	'''
	
	def __init__(self, eta = 0.01, n_iter = 10):
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
		self.errors_ = []
		
		for _ in range(self.n_iter):
			errors = 0
			for xi,target in zip(X,y):
				update = self.eta * (target - self.predict(xi))
				self.w_[1:] += update * xi 
				self.w_[0] += update
				errors += int(update != 0.0)
			self.errors_.append(errors)
		return self
		
		pass
		
	def net_input(self,X):
		return np.dot(X, self.w_[1:]) + self.w_[0] 
				#X does not contain the traditional x0 = 1 in front. 
				#So have to deal the first one separately.
		pass
	
	def predict(self, X):
		return np.where(self.net_input(X) > 0.0, 1, -1)
		pass
		
		
	pass
