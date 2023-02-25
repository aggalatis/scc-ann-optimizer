import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# Custom Classes
from classes.db import DB
from classes.customizer import Customizer

class WeightLogger:
    def __init__(self):
        self.log_weights()
    
    def log_weights(self):
        db = DB()
        cst = Customizer()    
        cst.load_optimal_model()
        loss = tf.keras.losses.BinaryCrossentropy()
        metrics = [
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]

        try:
            trainingData = np.array(db.get_all_table_data("training_data"))
            X = trainingData[:, 1:-1]
            Y = trainingData[:, -1]
            X = tf.keras.utils.normalize(X, axis=1)
            X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=cst.testSizes[0])
            ln = 1
            neuronsNumber = cst.neurons[0]
            model = tf.keras.Sequential()
            while ln <= cst.hiddenLayers[0]:
                if (ln == 1):
                    model.add(tf.keras.layers.Dense(cst.neurons[0], activation=cst.activators[0]))
                else:
                    neuronsNumber = round(neuronsNumber / 2)
                    model.add(tf.keras.layers.Dense(neuronsNumber, activation=cst.activators[0]))
                ln += 1
            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
            optim = cst.get_optimizer(cst.optimizers[0], cst.learningRates[0])
            model.compile(loss=loss, optimizer=optim, metrics=metrics)
            model.fit(x = X_train, y = Y_train, epochs = cst.epochs, batch_size=32, validation_data = (X_val, Y_val))
            f = open(os.getenv("WEIGHTS_FILE"), "w")
            for layerNum, layer in enumerate(model.layers):
                if layerNum != 0: continue
                weights = layer.get_weights()[0]
                # biases = layer.get_weights()[1]
                
                # for toNeuronNum, bias in enumerate(biases):
                #     f.write(f'{layerNum}Bias -> L{layerNum+1}N{toNeuronNum}: {bias}\n')
                full_weight_sum = 0
                for fromNeuronNum, wgt in enumerate(weights):
                    weights_summary = 0
                    for toNeuronNum, wgt2 in enumerate(wgt):
                        weights_summary = weights_summary + abs(wgt2 / 72)
                        f.write(f'L{layerNum}N{fromNeuronNum} -> L{layerNum+1}N{toNeuronNum} = {wgt2}\n')
                    full_weight_sum = full_weight_sum + weights_summary
                    f.write(f'Neuron {fromNeuronNum + 1} summary of weights: {str(weights_summary)}\n')
                f.write(f'All weights summary {full_weight_sum}')
            f.close()
            db.close_connection()
        except Exception as e:
            print(f"Log werights exception: {e}")
