import numpy as np
import math

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
    
    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def forward_propagate(self, inputs):

        activations = inputs

        for w in self.weights:
            #calculatate net input
            net_inputs = np.dot(activations, w)

            #calculate the activations
            activations = self._sigmoid(net_inputs)
        
        return activations

if __name__ == '__main__':

    #create a MLP
    mlp = MLP()
    #create input
    inputs = np.random.rand(mlp.num_imputs)
    #perform forward prop
    output = mlp.forward_propagate(inputs)
    # print output
    print('The input:',inputs)
    print('The output:',output)