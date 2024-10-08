import numpy as np

#Rectified Linear Unit

#for x >= 0, y = x
# for x < 0, y = 0
class Relu:
    def forward_pass(self, inputs):
        self.output = np.maximum(0, inputs)
        self.inputs = inputs
        
    def backwards_pass(self, dvalues):
        self.dinputs = dvalues.copy()
        
        self.dinputs[self.inputs <= 0] = 0