'Same code as we used to test our NeuralNetMLP implementation in Chap 12'

import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import pickle

#from MLPImplementation import NeuralNetMLP

def load_mnist(path, kind = 'train'):
    '''load mnist data from 'path' and save in numpy arrays.'''

    labels_path = os.path.join(path, "{:s}-labels-idx1-ubyte".format(kind))
    images_path = os.path.join(path, "{:s}-images-idx3-ubyte".format(kind))
    
    with open(labels_path, "rb") as lbpath:
        magic, n = struct.unpack(">II",lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    
    with open(images_path, "rb") as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels),784)
                    #images  are 28*28 -> 784 bytes
    
    print(labels.shape)
    print(images.shape)
    
    return images,labels
    
if __name__ == "__main__":
    print("Hello from MLPImplementation.")
    
    path = "../Chap12/Data"
    
    X_train, y_train = load_mnist(path, kind = "train")
    print("Rows: {:d} Columns: {:d}".format(X_train.shape[0], X_train.shape[1]))
    
    X_test, y_test = load_mnist(path, kind = "t10k")
    print("Rows: {:d} Columns: {:d}".format(X_test.shape[0], X_test.shape[1]))
    
    'Preparation of training data.'
    'casting the MNST image array into 32-bit format'
    import theano
    #theano.config.floatX = "float32"
    X_train = X_train.astype(theano.config.floatX)
    X_test = X_test.astype(theano.config.floatX)
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    
    'Convert the class labels in one-hot format'
    from keras.utils import np_utils
    print("First 3 lables: ", y_train[:3])
    y_train_ohe = np_utils.to_categorical(y_train)
    print("First 3 labels (one-hot):", y_train_ohe[:3])
    print("y_train_ohe shape:", y_train_ohe.shape)    

    'Now implement a neural network'
    from keras.models import Sequential
    from keras.layers.core import Dense
    from keras.optimizers import SGD
    
    np.random.seed(1)
    
    model = Sequential()    #Feedforward neural network
    
    model.add( Dense(input_dim = X_train.shape[1], output_dim = 50, init = "uniform", activation = "tanh") )
    
    model.add( Dense(input_dim = 50, output_dim = 50, init = "uniform", activation = "tanh") )
     
    model.add( Dense(input_dim = 50, output_dim = y_train_ohe.shape[1], init = "uniform", activation = "softmax") )
    
    sgd = SGD( lr = 0.001, decay = 1e-7, momentum = 0.9 )
        #stochastic gradient descent optimization
        #weight decay constant
        #momentum learning
        #learning rate
    
    model.compile(loss = "categorical_crossentropy", optimizer = sgd, metrics = ['accuracy'])
        #cost function set as cateorical_crossentropy
            #cross-entropy is the same as logistic regression
            #categorical_crossentropy is the multi-class version via softmax
            
    'model creating done, now train it'
    model.fit(X_train, y_train_ohe, epochs = 100, batch_size = 300, verbose = 1, validation_split = 0.1)
        # show_accuracy = True keyword is deprecated. In stead, need to use metrics = ['accuracy'] in the compile syntax.
    
    'save the model as a pickle'
    #pcklName = "NN_trained.p"
    
    #pickle.dump(model, open(pcklName,"wb"))
    #print("pickled as NN_trained.p")
    
    #read from pickle and do further processing
    #    print("Reading pickle ...")
    #    model = pickle.load(open(pcklName, "rb"))
    
    'Do some prediction. '
    y_train_pred = model.predict_classes(X_train, verbose = 0)
    print("First 3 predicitons:", y_train_pred[:3])
    
    'Finally, lets print the model accuracy on training and test sets'
    train_acc = np.sum(y_train == y_train_pred, axis = 0) / y_train.shape[0]
    print("training accuracy: {:.2f}".format(train_acc * 100))
    
    y_test_pred = model.predict_classes(X_test, verbose = 0)
    test_acc = np.sum(y_test == y_test_pred, axis = 0) / y_test.shape[0]
    print("test accuracy: {:.2f}".format(test_acc * 100))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
