import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import pickle

from MLPImplementation import NeuralNetMLP

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
    
    #print(labels.shape)
    #print(images.shape)
    
    return images,labels
    
if __name__ == "__main__":
    print("Hello from MLPImplementation.")
    
    X_train, y_train = load_mnist("Data", kind = "train")
    print("Rows: {:d} Columns: {:d}".format(X_train.shape[0], X_train.shape[1]))
    
    X_test, y_test = load_mnist("Data", kind = "t10k")
    print("Rows: {:d} Columns: {:d}".format(X_test.shape[0], X_test.shape[1]))
    
    'Visualize the data'
    fig,ax = plt.subplots(nrows = 2, ncols = 5, sharex = True, sharey = True)
    
    ax = ax.flatten()
    
    #imgArr = X_train[y_train == 2]       
    for i in range(10):
        img = X_train[y_train == i][0].reshape(28,28)
        #img = imgArr[i].reshape(28,28)
        ax[i].imshow(img, cmap = "Greys", interpolation = "nearest")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()
    

    'visualize the same digits from different sources'
    fig,ax = plt.subplots(nrows = 5, ncols = 5, sharex = True, sharey = True)
    ax = ax.flatten()
    digit = 7   #arbitrarily chosen
    for i in range(25):
        img = X_train[y_train == digit][i].reshape(28,28)
        ax[i].imshow(img, cmap = "Greys", interpolation = "nearest")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()
    

    
    #create the NN
    #    print("creating a 784-50-10 MLP ...")
    #    nn = NeuralNetMLP(n_output = 10, n_features = 784, n_hidden = 50, l2 = 0.1, l1 = 0.0, epochs = 1000, eta = 0.001, alpha = 0.001, decrease_const = 0.00001, shuffle = True, minibatches = 50, random_state = 1 )
    #    print("initialized.")
    #    
    
    #fit the NN
    #nn.fit(X_train, y_train, print_progress = True)
    #print("training done.")
    
    #once fitting is done, save as a pickle
    pcklName = "MLP_trained.p"
    
    #    pcklName = "MLP_trained.p"
    #    pickle.dump(nn, open(pcklName,"wb"))
    #    print("pickled as MLP_trained.p")
    #    #    
    #read from pickle and do further processing
    print("Reading pickle ...")
    nn = pickle.load(open(pcklName, "rb"))
    #    print(nn.n_features, nn.n_hidden, nn.n_output)
    #    print("cost:", nn.cost_)
    #    

    #plot the cost per mini-batch vs epoch
    #    plt.plot(range(len(nn.cost_)), nn.cost_)
    #    plt.ylim([0,2000])
    #    plt.ylabel("Cost")
    #    plt.xlabel("Epochs * 50")
    #    plt.tight_layout()
    #    plt.show()
    #    
    
    #make the plot smoother by taking average of each minibatch
    #    batches = np.array_split(range(len(nn.cost_)), 1000)
    #    cost_array = np.array(nn.cost_)
    #    cost_avgs = [np.mean(cost_array[i]) for i in batches]
    #    plt.plot(range(len(cost_avgs)), cost_avgs, color = "red")
    #    plt.ylim([0,2000])
    #    plt.ylabel("Cost")
    #    plt.xlabel("Epochs")
    #    plt.tight_layout()
    #    plt.show()
    #    
    
    #evaluate the performance of the model by calculating prdiction accuracy
    y_train_pred = nn.predict(X_train)
    acc = np.sum(y_train == y_train_pred, axis = 0) / X_train.shape[0]
    print("Training accuracy: {:,.2f}%".format(acc*100))
    
    y_test_pred = nn.predict(X_test)
    acc = np.sum(y_test == y_test_pred, axis = 0) / X_test.shape[0]
    print("Test accuracy: {:,.2f}%".format(acc*100))
    
    'some of the images where the NN struggle to identify'
    miscl_image = X_test[y_test != y_test_pred][:25]
    correct_label = y_test[y_test != y_test_pred][:25]
    miscl_label = y_test_pred[y_test != y_test_pred][:25]
    
    fig,ax = plt.subplots(nrows = 5, ncols = 5, sharex = True, sharey = True)
    
    ax = ax.flatten()
    
    for i in range(25):
        img = miscl_image[i].reshape(28,28)
        ax[i].imshow(img, cmap = "Greys", interpolation = "nearest")
        ax[i].set_title("{:,d}) t:{:,d} p:{:,d}".format(i+1, correct_label[i], miscl_label[i]))
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()
        
    
    'optional, if we want to save the data in csv format'
    #    np.savetxt(os.path.join("Data","train_img.csv"), X_train, fmt = "%i", delimiter = ',')
    #    np.savetxt(os.path.join("Data","train_labels.csv"), y_train, fmt = "%i", delimiter = ',')
    #    np.savetxt(os.path.join("Data","test_img.csv"), X_test, fmt = "%i", delimiter = ',')
    #    np.savetxt(os.path.join("Data","test_labels.csv"), y_test, fmt = "%i", delimiter = ',')
    
    'once the csvs are there, we can read and load data from them'
    #    X_train = np.genfromtxt(os.path.join("Data","train_img.csv"),dtype = int, delimiter = ',')
    #    y_train = np.genfromtxt(os.path.join("Data","train_labels.csv"),dtype = int, delimiter = ',')
    #    X_test = np.genfromtxt(os.path.join("Data","test_img.csv"),dtype = int, delimiter = ',')
    #    y_test = np.genfromtxt(os.path.join("Data","test_labels.csv"),dtype = int, delimiter = ',')
    #    print(X_train.shape)
    #    print(y_train.shape)
    #    print(X_test.shape)
    #    print(y_test.shape)
    #    
