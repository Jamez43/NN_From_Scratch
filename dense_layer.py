import numpy as np

class DenseLayer:
    
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward_pass(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        #remember for backpropagation
        self.inputs = inputs
        
    def backwards_pass(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dinputs = np.dot(dvalues, self.weights.T)
        self.dbias = np.sum(dvalues, axis=0, keepdims=True)