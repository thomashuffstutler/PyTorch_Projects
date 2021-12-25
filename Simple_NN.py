import numpy as np

class Model():

    def __init__(self, X, Y, weights_1, weights_2):
        self.X = X
        self.Y = Y
        self.weights_1 = weights_1
        self.weights_2 = weights_2
        
    def sigmoid(self, X):
        return 1/(1+np.exp(-self.X))

    def sigmoid_deriv(self):
        return self.layer_3*(1-self.layer_3)

    def update_step(self, layer, layer_error, tangent_dir):
        return self.layer_error*self.sigmoid_deriv(layer)

    def forward_pass(self):
        self.layer_1 = self.X
        self.layer_2 = self.sigmoid(np.dot(self.layer_1, self.weights_1))
        self.layer_3 = self.sigmoid(np.dot(self.layer_2.T, self.weights_2))

        self.layer_3_loss = self.Y - self.layer_3
        self.layer_3_change = self.layer_3_loss*self.sigmoid_deriv()
        
        self.layer_2_loss = self.layer_3_change.dot(self.weights_1)
        #self.layer_2_change = self.layer_2_loss*self.sigmoid_deriv(self.layer_2)

        self.weights_1 += self.layer_3_loss.T
        self.weights_2 += np.dot(self.layer_2_loss.T, self.weights_2)
        



epochs = 10

X = np.array([[1,0,0],
     [0,1,1],
     [1,1,0],
        [0,0,1]])

Y = np.array([[1],
              [1],
              [0],
              [0]])

np.random.seed(1)

weights_1 = 2*np.random.random((3,4)) - 1

weights_2 = 2*np.random.random((4,1)) - 1


NN = Model(X, Y, weights_1, weights_2)

for i in range(epochs):

    NN.forward_pass()

    if i <5:
        print(NN.layer_3_loss)

    
