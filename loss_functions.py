import numpy as np


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward_pass(output, y)
        
        data_loss = np.mean(sample_losses)
        
        return data_loss
    
class CategoricalCrossentropy(Loss):
    #all data will be clipped so we don't take the log of 0 or 1
    #-log(0) = infinity
    #-log(1) = 0
    def forward_pass(self, y_pred, y_true):
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

    def backwards_pass(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples