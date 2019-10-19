import theano
import numpy as np
from theano import tensor as T

#initialize
x = T.fmatrix("x")
#w = theano.shared(np.asarray([ [0.0, 0.0, 0.0]], dtype = theano.config.floatX ))
w = theano.shared(np.asarray([ [0.0 for ii in range(3)]], dtype = theano.config.floatX ))
z = x.dot(w.T)
update = [[w, w+1.0]]

#compile
net_input = theano.function(inputs = [x], updates = update, outputs = z, allow_input_downcast=True)

#execute
data = np.array([[1,2,3]], dtype = theano.config.floatX)

for i in range(5):
    print("z[{:2d}]: {:.2f}".format(i, float(net_input(data))))
    
    
