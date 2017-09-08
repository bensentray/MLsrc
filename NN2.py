import numpy as np
X = np.array([[0,0,1],
              [0,1,0],
              [0,1,1],
              [1,0,0],
              [1,0,1],
              [1,1,0],
              [1,1,1]]).transpose()
Y = np.array([[1],
              [1],
              [0],
              [0],
              [1],
              [1],
              [1]]).transpose()

# define sigmoid function
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
#define derivative of sigmoid function
def sigmoid_d(z):
    return sigmoid(z)*(1.0-sigmoid(z))

def crossEntropyCost(A,Y):
    '''calc Cost using cross entropy function'''
    '''The np.nan_to_num ensures that that is converted
        to the correct value (0.0)'''
    return np.sum(np.nan_to_num(-Y*np.log(A)-(1-Y)*np.log(1-A)))

class Network(object):
    def __init__(self,sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.weight_init()
    def weight_init(self):
        self.biases = [np.zeros((b,1),dtype=float) for b in self.sizes[1:]]
        self.weights = [np.random.randn(y,x)*0.01
                        for y,x in zip(self.sizes[1:],self.sizes[:-1])]

    def forward_progress(self,X):
        # feed forward,using sigmoid activation functions
        activation = X
        self.Zs = []        # list to store all the z vectors, layer by layer
        self.As = [X]       # list to store all the activations, layer by layer
        for b, w in zip(self.biases, self.weights):
            Z = np.dot(w, activation) + b
            activation = sigmoid(Z)
            self.Zs.append(Z)
            self.As.append(activation)

    def back_propagation(self,X,Y,lrate):
        self.d_b = [np.zeros(b.shape) for b in self.biases]
        self.d_w = [np.zeros(w.shape) for w in self.weights]

        m = X.shape[1]  #get number of the mini batch samples (X)

        # for last layer
        d_z = self.As[-1] - Y   # using cross entropy cost functions
        self.d_w[-1] = np.dot(d_z,self.As[-2].T)
        self.d_b[-1] = (1.0/m)*np.sum(d_z,axis=1,keepdims=True)
        # for other hidden layers
        for l in range(2, self.num_layers):
            d_z = np.dot(self.weights[-l + 1].T, d_z) * sigmoid_d(self.Zs[-l])
            self.d_w[-l] = np.dot(d_z, self.As[-l - 1].T)
            self.d_b[-l] = (1.0/m)*np.sum(d_z,axis=1,keepdims=True)

        # update the weights and biases
        self.weights = [w - lrate*nw for w, nw in zip(self.weights, self.d_w)]
        self.biases  = [b - lrate*nb for b, nb in zip(self.biases, self.d_b)]

#two layers network  :　hidden layer neural unit =n1,output layer neural unit = n2

if __name__ == '__main__' :
    np.random.seed(0)
    net = Network([3,5,1])
    for i in range(20000):
        net.forward_progress(X)
        net.back_propagation(X,Y,0.5)
        if i % 1000 == 0:
            print(crossEntropyCost(net.As[-1],Y))
    # print predict output
    print(net.As[-1])