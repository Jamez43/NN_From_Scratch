from relu import Relu
from softmax import Softmax
from dense_layer import DenseLayer
from loss_functions import CategoricalCrossentropy
from softmax_and_categorical_crossentropy import Softmax_CategoricalCrossentropy
from sgd import SGD
from adam import Adam
import numpy as np
import nnfs
from nnfs.datasets import spiral_data #nnfs = neurel networks from scratch(name of the book)
from matplotlib import pyplot as plt

nnfs.init()

def get_accuracy(output, y):
    prediction = np.argmax(output, axis=1)
    #convert from one-hot to sparse
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(prediction == y)
    return accuracy

#x shape =  (samples,2)
# Create dataset
X, y = spiral_data(samples=100, classes=3)
# Create Dense layer with 2 input features and 64 output values
dense1 = DenseLayer(2, 64)
# Create ReLU activation (to be used with Dense layer):
activation1 = Relu()
# Create second Dense layer with 64 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = DenseLayer(64, 3)
# Create Softmax classifier's combined loss and activation
loss_activation = Softmax_CategoricalCrossentropy()
# Create optimizer
optimizer = Adam(learning_rate=0.05, decay=5e-7)
# Train in loop
for epoch in range(10001):
    # Perform a forward pass of our training data through this layer
    dense1.forward_pass(X)
    # Perform a forward pass through activation function
    # takes the output of first dense layer here
    activation1.forward_pass(dense1.output)
    # Perform a forward pass through second Dense layer
    # takes outputs of activation function of first layer as inputs
    dense2.forward_pass(activation1.output)
    # Perform a forward pass through the activation/loss function
    # takes the output of second dense layer here and returns loss
    loss = loss_activation.forward_pass(dense2.output, y)
    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)
    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
        f'acc: {accuracy:.3f}, ' +
        f'loss: {loss:.3f}, ' +
        f'lr: {optimizer.current_learning_rate}')
    # Backward pass
    loss_activation.backwards_pass(loss_activation.output, y)
    dense2.backwards_pass(loss_activation.dinputs)
    activation1.backwards_pass(dense2.dinputs)
    dense1.backwards_pass(activation1.dinputs)
    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()