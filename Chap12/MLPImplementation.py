import numpy as np
from scipy.special import expit 
import sys

class NeuralNetMLP(object):
    'creates a NN with one hidden layer'
    def __init__(self, n_output, n_features, n_hidden = 30, l1 = 0.0, l2 = 0.0, epochs = 500, eta = 0.001, alpha = 0.0, decrease_const = 0.0, shuffle = True, minibatches = 1, random_state = None):
        
        np.random.seed(random_state)
        self.n_output = n_output        #no of nodes in output layer
        self.n_features = n_features    #no of nodes in input layer
        self.n_hidden = n_hidden        #no of nodes in the only hidden layer
        self.w1, self.w2 = self._initialize_weights()
        self.l1 = l1                    #lambda parameter of the l1 regularization
        self.l2 = l2                    #lambda parameter of the l2 regularization
        self.epochs = epochs            #the number of passes over the training set
                
        self.eta = eta                  #the learning rate
        self.alpha = alpha              #a parameter for momentum learning to add a factor of the previous gradient to the weight update for faster learning
                                        # DW_t = eta * DJ(W_t) + alpha * DW_(t-1)
                                            #here t is the current time or epoch
        self.decrease_const = decrease_const
                                        #the decreae constant d for an adaptive learning rate eta that decreases over time for better convergence: eta/(1+t*d)
        self.shuffle = shuffle          #shuffling the training set prior to every epoch to prevent the algorithm from getting stuck in cycles
        
        self.minibatches = minibatches  #Splitting the training data into k mini-batches in each epoch. The gradient is computed for each mini-batch separately instead of the entire training data for faster learning.
        pass
        
    '''
        the steps a NN goes through:
            1. Randomly initialize the weights
            2. Implement forward propagation to get h_theta(x) for any x
            3. Compute the cost through cost function J(theta)
            4. Implement back-propagation to compute partial derivative of J(theta)
        
        In our code, insted of theta we write W1 and W2. So,
            
        During forward propagation:
            A^1 = X
            Z^2 = W^1 * A^1
            A^2 = g(Z^2) 
            Z^3 = W^2 * A^2
            A^3 = g(Z^3)
                = H_theta(X) 
       
        Now do back propagation:
            Error at layer 3: sigma^3 = A^3 - y
            
            Error at layer 2: sigma^2 = (Theta^2)_T * sigma^3 .* A^2 .* (1 - A^2)            
            
            No error at the input layer.
            
            Finally, for each training data: D^l := D^l + gradient^l = D^l + A^l * sigma^(l+1)
            
            So we get D^1, D^2 this way. 
            
            In our code:
                gradient^1 = sigma^2 * a^1
                gradient^2 = sigma^3 * a^2
            
            As matrix: D^l = D^l + sigma^(l+1) * (A^l)_T
            
        After all the training data is used, we get our desired derivative of the cost function, D = 1/m * D
        
        The use gradient descent to minimize the cost function.
    '''
        
    def _encode_labels(self, y, k):
        
        onehot = np.zeros((k, y.shape[0]))
        for idx, val in enumerate(y):
            onehot[val, idx] = 1.0
        return onehot
        
    def _initialize_weights(self):
        w1 = np.random.uniform(-1.0, 1.0, size = self.n_hidden * (self.n_features + 1))
        w1 = w1.reshape(self.n_hidden, self.n_features + 1)
        
        w2 = np.random.uniform(-1.0, 1.0, size = self.n_output * (self.n_hidden + 1) )
        w2 = w2.reshape(self.n_output, self.n_hidden + 1)
        
        return w1, w2
        
    def _sigmoid(self, z):
        #expit is equivalent to 1.0 / (1.0 + np.exp(-z))
        #return expit(z)
        return 1.0 / (1.0 + np.exp(-z))
        
    def _sigmoid_gradient(self, z): 
        sg = self._sigmoid(z)
        return sg * (1 - sg)    #remember, Andrew's course had a magic jump that said
                                #with calculas, it can be shown that,
                                #if a3 = g(z3); then its derivative, g`(z3) = a3 * (1-a3)
                                
                                
    def _add_bias_unit(self, X, how = "column"):
        if how == "column":
            X_new = np.ones( (X.shape[0], X.shape[1]+1) )
            X_new[:,1:] = X
            
        elif how == "row":
            X_new = np.ones( (X.shape[0]+1, X.shape[1]) )
            X_new[1:,:] = X
        else:
            raise AttributeError("'how' must be 'column' or 'row'")
        return X_new
        
    def _feedforward(self, X, w1, w2):
        #because we know there are only one hidden layer, it's easy to calculate
        #need to calculate and return a1, z2, a2, z3, a3
        a1 = self._add_bias_unit(X, how = "column")
        z2 = w1.dot(a1.T)
        a2 = self._sigmoid(z2)
        a2 = self._add_bias_unit(a2, how = "row") #why row? Probably becasue it's already in transpose form.
        z3 = w2.dot(a2)
        a3 = self._sigmoid(z3)
        
        return a1, z2, a2, z3, a3
        
    #TODO: read-up L1 and L2 regularization again
    def _L2_reg(self, lambda_, w1, w2):
        return (lambda_/2.0) * (np.sum( w1[:,1:] ** 2) + np.sum(w2[:,1:] ** 2))
        
    
    def _L1_reg(self, lambda_, w1, w2):
        return (lambda_/2.0) * ( np.abs( w1[:,1:] ).sum()  + np.abs(w2[:,1:]).sum() )
        
    def _get_cost(self, y_enc, output, w1, w2):
        term1 = -y_enc * (np.log(output))
        term2 = (1-y_enc) * np.log(1 - output)
        cost = np.sum(term1 - term2)
        L1_term = self._L1_reg(self.l1, w1, w2) 
        L2_term = self._L2_reg(self.l2, w1, w2)
        cost = cost + L1_term + L2_term
        #print(w2)
        return cost
        
    def _get_gradient(self, a1, a2, a3, z2, y_enc, w1, w2):
        #backpropagation
        sigma3 = a3 - y_enc
        z2 = self._add_bias_unit(z2, how = "row")
        sigma2 = w2.T.dot(sigma3) * self._sigmoid_gradient(z2)
        sigma2 = sigma2[1:, :]
        grad1 = sigma2.dot(a1)
        grad2 = sigma3.dot(a2.T)
        
        #regularize
        grad1[:, 1:] += ( w1[:, 1:] * (self.l1 + self.l2) )
        grad2[:, 1:] += ( w2[:, 1:] * (self.l1 + self.l2) )
         
        return grad1, grad2
        
    def predict(self, X):
        a1, z2, a2, z3, a3 = self._feedforward(X, self.w1, self.w2)
        y_pred = np.argmax(z3, axis = 0)
        return y_pred
        
    def fit(self, X, y, print_progress = False):
        
        #get the containters: cost, w1, w2
        self.cost_ = []
        delta_w1_prev = np.zeros(self.w1.shape)
        delta_w2_prev = np.zeros(self.w2.shape)
        
        #get X and y
        X_data, y_data = X.copy(), y.copy()
        
        #encode the labels of y using one-hot encoding
        y_enc = self._encode_labels(y, self.n_output)   
        
        #run a loop epoch times
        for i in range(self.epochs):
            #set learning rate, shuffle data, get minibatches etc
            
            #adaptive learning rate
            self.eta /= (1 + self.decrease_const*i)
            
            if print_progress:
                sys.stderr.write("Epoch: {}/{}\n".format(i+1, self.epochs))
                sys.stderr.flush()
                
            if self.shuffle:
                idx = np.random.permutation(y_data.shape[0])
                X_data, y_data = X_data[idx], y_data[idx]
                
            mini = np.array_split(range(y_data.shape[0]), self.minibatches)
                
            #for each minibatch do the follwing:
            for ii,idx in enumerate(mini):
            
                # do forward propagation
                    # get a1, z2, a2, z3, a3
                a1, z2, a2, z3, a3 = self._feedforward(X[idx], self.w1, self.w2) 
                # calculate cost
                cost = self._get_cost(y_enc = y_enc[:, idx], output = a3, w1 = self.w1, w2 = self.w2)   
                
                self.cost_.append(cost)
                print("cost:",cost)
                        
                # do backpropagation. 
                    #get grad1, grad2
                grad1, grad2 = self._get_gradient(a1 = a1, a2 = a2, a3 = a3, z2 = z2, y_enc = y_enc[:, idx], w1 = self.w1, w2 = self.w2)    
                    
                # update weights
                delta_w1, delta_w2 = self.eta * grad1, self.eta * grad2
                self.w1 = self.w1 - (delta_w1 + (self.alpha *delta_w1_prev))
                self.w2 = self.w2 - (delta_w2 + (self.alpha *delta_w2_prev))
                delta_w1_prev, delta_w2_prev = delta_w1, delta_w2 
                
                #if ii == 0:
                #    break
        #Effectively, each mini-batch is processed in a matrix, so much faster than looping over training dataset one by one.
        
        #Each mini-batch goes in a loop one after another, doing the learning.
        
        return self
        
        'Go through the gradient descent alorithm once again. See how and why weights are updated.'   
        
    pass
    
    
if __name__ == "__main__":
    #initialize a new 784-50-10 MLP
    print("creating a 784-50-10 MLP ...")
    nn = NeuralNetMLP(n_output = 10, n_features = 784, n_hidden = 50, l2 = 0.1, l1 = 0.0, epochs = 1000, eta = 0.001, alpha = 0.001, decrease_const = 0.00001, shuffle = True, minibatches = 50, random_state = 1 )
    print("initialized.")
    pass
