import numpy as np


#4 input nodes, 3 output nodes (4 inputs, 3 sets of weights & biases)

#(3,4)
inputs = [[1.0,2.0,3.0,2.5],
          [2.0,5.0,-1.0,2.0],
          [-1.5,2.7,3.3,-0.8]]
#(3, 4)
#(4,3) Transposed
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0,3.0,0.5]

layer_outputs = np.dot(inputs, np.array(weights).T) + biases

print(layer_outputs)