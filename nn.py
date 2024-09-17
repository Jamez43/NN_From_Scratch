from relu import Relu
from softmax import Softmax
from dense_layer import DenseLayer
from loss_functions import Loss_CategoricalCrossentropy
import numpy as np
import nnfs
from nnfs.datasets import spiral_data #nnfs = neurel networks from scratch(name of the book)

nnfs.init()

def get_accuracy(output, y):
    prediction = np.argmax(output, axis=1)
    #convert from one-hot to sparse
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(prediction == y)
    return accuracy

#x shape =  (samples,2)
x, y = spiral_data(samples=100, classes=3)

dense1 = DenseLayer(2, 3)
activation1 = Relu()
dense1.forward_pass(x)
activation1.forward(dense1.output)


dense2 = DenseLayer(3, 3)
activation2 = Softmax()
dense2.forward_pass(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

loss_function = Loss_CategoricalCrossentropy()

loss = loss_function.calculate(activation2.output, y)

print('Loss:', loss)
print('Accuracy:', get_accuracy(activation2.output, y))