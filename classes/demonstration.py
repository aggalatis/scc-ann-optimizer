import tensorflow as tf
# import keras_tuner as kt
import numpy as np
from sklearn.model_selection import train_test_split
from classes.plots import Plot

# Custom Classes
from classes.db import DB


class Demonstration:
    def __init__(self):
        # Random Example
        self.TEST_SIZE = 0.10
        self.ACTIVATOR = 'sigmoid'
        self.EPOCHS = 100
        self.LEARNING_RATE = 0.002
        self.PLOT_NAME = 'demonstration'
        self.OPTIMIZER = tf.keras.optimizers.Adagrad(learning_rate=self.LEARNING_RATE)
        # self.train_model()

        # Optimal Modal
        self.OPTIMAL_TEST_SIZE = 0.12
        self.OPTIMAL_ACTIVATOR = 'relu'
        self.OPTIMAL_EPOCHS = 100
        self.OPTIMAL_LEARNING_RATE = 0.004
        self.OPTIMAL_PLOT_NAME = 'selected'
        self.OPTIMAL_OPTIMIZER = tf.keras.optimizers.Adamax(learning_rate=self.OPTIMAL_LEARNING_RATE)
        self.train_optimal_model()
    
    def train_model(self):
        db = DB()
        plot = Plot() 
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
            X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=self.TEST_SIZE)
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Dense(72, activation=self.ACTIVATOR))       
            model.add(tf.keras.layers.Dense(36, activation=self.ACTIVATOR))       
            model.add(tf.keras.layers.Dense(18, activation=self.ACTIVATOR))       
            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
            model.compile(loss=loss, optimizer=self.OPTIMIZER, metrics=metrics)
            history = model.fit(x = X_train, y = Y_train, epochs = self.EPOCHS, batch_size=32, validation_data = (X_val, Y_val))
            model.summary()
            plot.save_plot(history.history['loss'], history.history['val_loss'], self.PLOT_NAME, 'loss')
            plot.save_plot(history.history['binary_accuracy'], history.history['val_binary_accuracy'], self.PLOT_NAME, 'accuracy')

            testData = np.array(db.get_all_test_data("test_data"))
            x_test = testData[: , 1:-1]
            test_ids = testData[:, 0]
            y_test = testData[:, -1]
            x_test= tf.keras.utils.normalize(x_test, axis=1)
            predictions = model.predict(x_test)
            loss_counter = 0
            wrong_ids = ""
            for i in range(predictions.size):
                predictedClass = 1
                if (predictions[i] < 0.5): predictedClass = 0
                if (predictedClass != y_test[i]):
                    loss_counter += 1
                    wrong_ids = wrong_ids + "|" + str(test_ids[i])
            final_acc =  (1 -  loss_counter / predictions.size)
            wrong_ids = wrong_ids[1:]
            print(f"Max.Acc: {max(history.history['val_binary_accuracy'])}, Min.Acc: {min(history.history['val_binary_accuracy'])} Max.Loss: {max(history.history['val_loss'])} Min.Loss: {min(history.history['val_loss'])}, Final Acc: {final_acc}, Loss Count: {loss_counter}, Wrong IDS: {wrong_ids} \n")
            # output = {"maxAcc": max(history.history['val_binary_accuracy']), "minAcc": min(history.history['val_binary_accuracy']), "maxLoss": max(history.history['val_loss']), "minLoss": min(history.history['val_loss']), "jsonModel": model.to_json(), "epochs": cst.epochs, "finalAcc": final_acc, "lossCount": f"{loss_counter}", "wrongIds": wrong_ids}
            # db.save_model(output)
            db.close_connection()
        except Exception as e:
            print(f"Main exception: {e}")

    
    def train_optimal_model(self):
        db = DB()
        plot = Plot() 
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
            X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.12)
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Dense(72, activation='relu'))       
            model.add(tf.keras.layers.Dense(36, activation='relu'))       
            model.add(tf.keras.layers.Dense(18, activation='relu'))       
            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
            model.compile(loss=loss, optimizer=tf.keras.optimizers.Adamax(learning_rate=0.004), metrics=metrics)
            history = model.fit(x = X_train, y = Y_train, epochs = 100, batch_size=32, validation_data = (X_val, Y_val))
            model.summary()
            # plot.save_plot(history.history['loss'], history.history['val_loss'], 'loss_plot', 'loss')
            # plot.save_plot(history.history['binary_accuracy'], history.history['val_binary_accuracy'], 'acc_plot', 'accuracy')

            testData = np.array(db.get_all_test_data("test_data"))
            x_test = testData[: , 1:-1]
            test_ids = testData[:, 0]
            y_test = testData[:, -1]
            x_test= tf.keras.utils.normalize(x_test, axis=1)
            predictions = model.predict(x_test)
            wrong_prediction_counter = 0
            wrong_ids = ""
            for i in range(predictions.size):
                predictedClass = 1
                print(f"ID: {test_ids[i]} -> {predictions[i]}")
                if (predictions[i] < 0.5): predictedClass = 0
                if (predictedClass != y_test[i]):
                    wrong_prediction_counter += 1
                    wrong_ids = wrong_ids + "|" + str(test_ids[i])
            final_acc =  (1 -  wrong_prediction_counter / predictions.size)
            wrong_ids = wrong_ids[1:]
            print(f"Max. Acc {max(history.history['val_binary_accuracy'])}")
            print(f"Min.Acc: {min(history.history['val_binary_accuracy'])}")
            print(f"Max.Loss: {max(history.history['val_loss'])}")
            print(f"Min.Loss: {min(history.history['val_loss'])}")
            print(f"Final Acc: {final_acc}")
            print(f"Wrong Predictions: {wrong_prediction_counter}")
            print(f"Wrong IDS: {wrong_ids}")
            db.close_connection()
        except Exception as e:
            print(f"Exception: {e}")