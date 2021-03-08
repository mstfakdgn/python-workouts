import numpy as np
import math

# save activations and derivatives
# implement backpropagation
# implement gradient descent
# implement train
# train our net with some dummy dataset
# make some predictions

class MLP:

    def __init__(self, num_imputs = 3, num_hidden = [3,5], num_outputs = 2):
        self.num_imputs = num_imputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers = [self.num_imputs] + self.num_hidden + [self.num_outputs]

        #initiate random weights
        self.weights = []

        for i in range(len(layers) -1):
            w = np.random.rand(layers[i], layers[i+1])
            self.weights.append(w)

        activations = []

        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        
        self.activations = activations

        derivatives = []

        for i in range(len(layers)-1):
            d = np.zeros((layers[i], layers[i+1]))
            derivatives.append(d)
        
        self.derivatives = derivatives
    
    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def forward_propagate(self, inputs):

        activations = inputs
        self.activations[0] = inputs

        for w in self.weights:
            #calculatate net input
            net_inputs = np.dot(activations, w)

            #calculate the activations
            activations = self._sigmoid(net_inputs)
            self.activations[i+1] = activations
        
        return activations
    
    def back_propagate(self, error):

        # dE/dW_i = (y - a_[i+1]) s'(h_[i+1]) a_i
        # s(h_[i+1]) = s(h_[i+1])(1 - s(h_[i+1]))
        # s(h_[i+1]) = a_[i+1]
        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]
            delta = error * self._sigmoid_derivative(activations)

if __name__ == '__main__':

    #create a MLP
    mlp = MLP()
    #create input
    np.random.seed(10)
    inputs = np.random.rand(mlp.num_imputs)
    #perform forward prop
    output = mlp.forward_propagate(inputs)
    # print output
    print('The input:',inputs)
    print('The output:',output)