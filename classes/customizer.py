import numpy as np
import tensorflow as tf
class Customizer:
    def __init__(self):
        self.activators = ['relu', 'sigmoid', 'softmax', 'softsign', 'tanh', 'selu', 'elu', 'exponential']
        self.hiddenLayers = [1, 2, 3]
        self.neurons = [72, 36, 18]
        self.optimizers = ['adam', 'Adamax', 'Nadam', 'RMSprop', 'SGD', 'Adadelta', 'Adagrad']
        self.epochs = 100
        self.learningRates = [0.002, 0.003, 0.004, 0.005]
        self.testSizes = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]

    def min_max_normalization(self, data):
        for c in range(data.shape[1]):
            x_col = data[:, c]
            max_value = np.max(x_col)
            min_value = np.min(x_col)
            data[:, c] = (x_col - min_value) / (max_value-min_value)
        return data
    
    def mean_std_normalization(self, data):
        for c in range(data.shape[1]):
            x_col = data[:, c]
            mean_value = np.mean(x_col)
            std_value = np.std(x_col)
            data[:, c] = (x_col - mean_value) / std_value
        return data

    def get_optimizer(self, optimizerName, learningRate):
        if (optimizerName == 'adam'): return tf.keras.optimizers.Adam(learning_rate=learningRate)
        if (optimizerName == 'Adamax'): return tf.keras.optimizers.Adamax(learning_rate=learningRate)
        if (optimizerName == 'Nadam'): return tf.keras.optimizers.Nadam(learning_rate=learningRate)
        if (optimizerName == 'RMSprop'): return tf.keras.optimizers.RMSprop(learning_rate=learningRate)
        if (optimizerName == 'SGD'): return tf.keras.optimizers.SGD(learning_rate=learningRate)
        if (optimizerName == 'Adadelta'): return tf.keras.optimizers.Adadelta(learning_rate=learningRate)
        if (optimizerName == 'Adagrad'): return tf.keras.optimizers.Adagrad(learning_rate=learningRate)
        return None
            

        
