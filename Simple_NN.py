#This is our linear algebra library
import numpy as np

#We create a class model with all of the operations we need
#This choice was made for simplicity in the end product (to make it more user friendly to run the model)
class Model():

    #This is our constructor where we initialize our variables:
    #X = inputs
    #Y = target output
    #weights_1 = first layer of weights in our neural network
    #weights_2 = second layer of weights in our neural network
    def __init__(self, X, Y, weights_1, weights_2):
        self.X = X
        self.Y = Y
        self.weights_1 = weights_1
        self.weights_2 = weights_2

    #We will use the sigmoid function here as our nonlinear activation function
    #This will give us outputs between 0 and 1
    def sigmoid(self, X):
        return 1/(1+np.exp(-self.X))

    #The derivative of the sigmoid will be used to calculate the change needed for our weights
    def sigmoid_deriv(self):
        return self.layer_3*(1-self.layer_3)

    #This update step function is optional (not used in the forward pass)
    #It could be used in forward pass for simplification
    def update_step(self, layer, layer_error, tangent_dir):
        return self.layer_error*self.sigmoid_deriv(layer)

    #In forward pass, we do our matrix multiplication between each layer of our weights and the corresponding previous layer of inputs
    #We also update our weight matrices at the end of this function
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
        


#This is the number of iterations which we will use to train our model
epochs = 10000

#This is our input array (shape is (4,3))
X = np.array([[1,0,0],
     [0,1,1],
     [1,1,0],
        [0,0,1]])

#This is our target output array (shape is (4,1))
Y = np.array([[1],
              [1],
              [0],
              [0]])

np.random.seed(1)

#These are our two matrices which are initialized with random numbers that have a mean of 1
weights_1 = 2*np.random.random((3,4)) - 1

weights_2 = 2*np.random.random((4,1)) - 1

#Here we create an instance of our model which we call NN for neural network
#We instantiate our variables in our constructor with the previously defined values as described above
NN = Model(X, Y, weights_1, weights_2)

#Here we run through our training loop
for i in range(epochs):

    #We run through a forward pass and update of our network at each iteration
    NN.forward_pass()

    '''
    This is an example of how to view the loss inour network
    if i <5:
        print(f'layer 2 loss is: {NN.layer_2_loss}')
    '''

    
