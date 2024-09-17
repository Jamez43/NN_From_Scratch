import numpy as np

#test data
softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])
class_targets = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 1, 0]])

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        
        data_loss = np.mean(sample_losses)
        
        return data_loss
    
class Loss_CategoricalCrossentropy(Loss):
    #all data will be clipped so we don't take the log of 0 or 1
    #-log(0) = infinity
    #-log(1) = 0
    def forward(self, y_pred, y_true):
        #y_pred is 2d
        batch = len(y_pred)
        
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        if len(y_true.shape) == 1:
            #first parameter is rows, second is columns
            highest_output_confidence = y_pred_clipped[range(batch), y_true]
        elif len(y_true.shape) == 2:
            #ex: [0.7, 0.1, 0.2] * [1, 0, 0] = [0.7, 0, 0]
            # sum = 0.7 + 0 + 0
            highest_output_confidence = np.sum(y_pred_clipped * y_true, axis=1)
        
        negative_log = -np.log(highest_output_confidence)
        return negative_log