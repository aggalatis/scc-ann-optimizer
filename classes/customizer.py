class Customizer:
    def __init__(self):
        self.activators = ['relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'selu']
        self.hiddenLayers = [1, 2, 3, 4]
        self.neurons = [144, 72, 36, 18]
        self.optimizers = ['adam', 'Adamax', 'Nadam', 'RMSprop', 'SGD', 'Adadelta', 'Adagrad']
        self.epochs = 100