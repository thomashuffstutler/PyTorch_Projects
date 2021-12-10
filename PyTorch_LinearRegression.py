import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

#Here we create our x data for our linear equation; this is our independent variable
#Note that we have to manually set the data type as np.float32 since otherwise they are generated as int's which will cause errors
#With our gradient calculations in our training loop
x_vals = [i for i in range(10)]
x_training = np.array(x_vals, dtype=np.float32)
x_training = x_training.reshape(-1,1)

#Here we create the y values based on our equation y = 3x + 1 for our generated values of x
y_vals = [3*i + 1 for i in x_vals]
y_training = np.array(y_vals, dtype=np.float32)
y_training = y_training.reshape(-1,1)

#Here we create our LinearRegression class which inherits from the pytorch nn module
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(1,1) #input size 1 output size 1

    #The forward function is where our operations are done
    def forward(self, X):
        out = self.linear(X)
        return out

#epochs is the number of iterations we will train our model for
#;earning rate is for our gradient parameter
epochs = 100
learning_rate = 0.01

#Here we create an instance of the LinearRegression class 
model = LinearRegression()

#If cuda is available, we want to use it so our program will run on the GPU
'''
if cuda.is_available():
    model.cuda()
'''

#Here, criterion specifies our loss function which is mean squared error (the standard loss function for linear regression)
#Also, optimizer specifies our gradient descent method (optimizer) which is stochastic gradient descent
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#Here, we turn our x and y training data from numpy arrays to pytorch tensors so they can be used with the pytorch methods
x_train = Variable(torch.from_numpy(x_training))
y_train = Variable(torch.from_numpy(y_training))

#We run our loop for the specified number of iterations and train our model
for epoch in range(epochs):

    #We want to se all of our gradients to zero, otherwise they will accumulate which we don't want
    optimizer.zero_grad()

    #This gives us our predicted values for each loop given our inout data
    pred_y = model(x_train)

    #This calculates the loss; MSE between our predicted value and actual training values
    loss = criterion(pred_y, y_train)

    #Backpropogation happens here with the call to loss.backwards which computes the derivatives of our prediction with respect to our paramters (weights and biases)
    loss.backward()

    #Then we apply the updates with our gradients calculated in the previous step
    optimizer.step()

    #We print the current iteration and the loss so that we can monitor the learning of our system
    print('epoch {}, loss{}'.format(epoch, loss.item()))

#Here, we test our model and get the predicted values and turn them into a numpy array so we can plot them for easier comprehension
predicted = model(x_train).data.numpy()

#This will plot the original x and y values in blue dots, as well as our predicted values vs original x data in a dotted orange line so that we may see the performance
plt.clf()
plt.plot(x_vals, y_vals, 'o',  alpha=0.5)
plt.plot(x_vals, predicted, '--', alpha=0.5)
plt.show()




    
