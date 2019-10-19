import theano
import numpy as np
from theano import tensor as T

#initialize
x = T.fmatrix(name = "x")
x_sum = T.sum(x, axis = 0)

#compile
calc_sum = theano.function(inputs = [x], outputs = x_sum, allow_input_downcast=True)    #"allow_input_downcast" is needed in 32-bit systems

#execute (Python list)
ary = [ [1,2,3],[1,2,3] ]
print("Col sum:", calc_sum(ary))

#execute (Numpy array)
ary = np.array([ [1,2,3],[1,2,3] ], dtype = theano.config.floatX)
print("Col sum:", calc_sum(ary))

