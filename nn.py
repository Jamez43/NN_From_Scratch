from relu import Relu
from softmax import Softmax
from dense_layer import DenseLayer
from loss_functions import CategoricalCrossentropy
from softmax_and_categorical_crossentropy import Softmax_CategoricalCrossentropy
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
activation1.forward_pass(dense1.output)


dense2 = DenseLayer(3, 3)
activation2 = Softmax()
dense2.forward_pass(activation1.output)

softmax_loss = Softmax_CategoricalCrossentropy()
softmax_loss.forward_pass(dense2.output, y)

print(softmax_loss.output[:5])
print("Loss:", softmax_loss.forward_pass(dense2.output,y))

print('Accuracy:', get_accuracy(softmax_loss.output, y))


#backwards
softmax_loss.backwards_pass(softmax_loss.output, y)
dense2.backwards_pass(softmax_loss.dinputs)
activation1.backwards_pass(dense2.dinputs)
dense1.backwards_pass(activation1.dinputs)

print(dense1.dweights)
print()
print(dense1.dbias)
print()
print(dense2.dweights)
print()
print(dense2.dbias)