from relu import Relu
from sigmoid import Sigmoid
from dense_layer import DenseLayer
from loss_functions import Loss_BinaryCrossentropy
from adam import Adam
import numpy as np
import nnfs
from nnfs.datasets import spiral_data #nnfs = neurel networks from scratch(name of the book)
from matplotlib import pyplot as plt

nnfs.init()

X, y = spiral_data(samples=100, classes=2)
# Reshape labels to be a list of lists
# Inner list contains one output (either 0 or 1)
# per each output neuron, 1 in this case
y = y.reshape(-1, 1)
# Create Dense layer with 2 input features and 64 output values
dense1 = DenseLayer(2, 64)
# Create ReLU activation (to be used with Dense layer):
activation1 = Relu()
# Create second Dense layer with 64 input features (as we take output
# of previous layer here) and 1 output value
dense2 = DenseLayer(64, 1)
# Create Sigmoid activation:
activation2 = Sigmoid()
# Create loss function
loss_function = Loss_BinaryCrossentropy()
# Create optimizer
optimizer = Adam(decay=5e-7)
# Train in loop
for epoch in range(10001):
    # Perform a forward pass of our training data through this layer
    dense1.forward_pass(X)
    # Perform a forward pass through activation function
    # takes the output of first dense layer here
    activation1.forward_pass(dense1.output)
    # Perform a forward pass through second Dense layer
    # takes outputs of activation function
    # of first layer as inputs
    dense2.forward_pass(activation1.output)
    # Perform a forward pass through activation function
    # takes the output of second dense layer here
    activation2.forward_pass(dense2.output)
    # Calculate the data loss
    loss = loss_function.calculate(activation2.output, y)

    # Calculate accuracy from output of activation2 and targets
    # Part in the brackets returns a binary mask - array consisting
    # of True/False values, multiplying it by 1 changes it into array
    # of 1s and 0s
    predictions = (activation2.output > 0.5) * 1
    accuracy = np.mean(predictions==y)
    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
            f'acc: {accuracy:.3f}, '+
            f'loss: {loss:.3f} ' +
            f'lr: {optimizer.current_learning_rate}')
    # Backward pass
    loss_function.backwards_pass(activation2.output, y)
    activation2.backwards_pass(loss_function.dinputs)
    dense2.backwards_pass(activation2.dinputs)
    activation1.backwards_pass(dense2.dinputs)
    dense1.backwards_pass(activation1.dinputs)
    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
    

# Validate the model
# Create test dataset
X_test, y_test = spiral_data(samples=100, classes=2)
# Reshape labels to be a list of lists
# Inner list contains one output (either 0 or 1)
# per each output neuron, 1 in this case
y_test = y_test.reshape(-1, 1)

# Perform a forward pass of our testing data through this layer
dense1.forward_pass(X_test)
# Perform a forward pass through activation function
# takes the output of first dense layer here
activation1.forward_pass(dense1.output)
# Perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward_pass(activation1.output)
# Perform a forward pass through activation function
# takes the output of second dense layer here
activation2.forward_pass(dense2.output)
# Calculate the data loss
loss = loss_function.calculate(activation2.output, y_test)
# Calculate accuracy from output of activation2 and targets
# Part in the brackets returns a binary mask - array consisting of
# True/False values, multiplying it by 1 changes it into array
# of 1s and 0s
predictions = (activation2.output > 0.5) * 1
accuracy = np.mean(predictions==y_test)
print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')