import numpy as np

#Rectified Linear Unit

#for x >= 0, y = x
# for x < 0, y = 0
class relu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        