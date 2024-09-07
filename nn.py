from relu import relu
from softmax import softmax
from dense_layer import dense_layer
import nnfs
from nnfs import datasets #nnfs = neurel networks from scratch(name of the book)

nnfs.init()

#x shape =  (samples,2)
x, y = datasets.spiral_data(samples=100, classes=3)

dense1 = dense_layer(2, 3)
activation1 = relu()
dense1.forward_pass(x)
activation1.forward(dense1.output)


dense2 = dense_layer(3, 3)
activation2 = softmax()
dense2.forward_pass(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])