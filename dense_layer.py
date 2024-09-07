import numpy as np

class dense_layer:
    
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward_pass(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases