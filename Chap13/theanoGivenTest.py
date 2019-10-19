'''
Run using: THEANO_FLAGS=floatX=float32 python3 theanoGivenTest.py 

'''

import theano
import numpy as np
from theano import tensor as T

#initialize
data = np.array([[1,2,3]], dtype = theano.config.floatX)
x = T.fmatrix("x")
#w = theano.shared(np.asarray([ [0.0, 0.0, 0.0]], dtype = theano.config.floatX ))
w = theano.shared(np.asarray([ [0.0 for ii in range(3)]], dtype = theano.config.floatX ))
z = x.dot(w.T)
update = [[w, w+1.0]]

#compile
net_input = theano.function(inputs = [], updates = update, givens = {x: data}, outputs = z)

#execute
for i in range(5):
    print("z[{:2d}]: {:.2f}".format(i, float(net_input())))
    
    
