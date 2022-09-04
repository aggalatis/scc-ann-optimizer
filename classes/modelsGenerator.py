import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# Custom Classes
from classes.db import DB
from classes.customizer import Customizer

class ModelsGenerator:
    def __init__(self):
        self.generate_models()
    
    def generate_models(self, deleteOldModels=False):
        db = DB()
        cst = Customizer()    
        loss = tf.keras.losses.BinaryCrossentropy()
        metrics = [
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
        if (deleteOldModels == True):
            db.truncate_table("models")

        for learning_rate in cst.learningRates:
            for test_size in cst.testSizes:
                for opt in cst.optimizers:
                    for act in cst.activators:
                        for neurnum in cst.neurons:
                            for layerCount in cst.hiddenLayers:
                                try:
                                    trainingData = np.array(db.get_all_table_data("training_data"))
                                    X = trainingData[:, 1:-1]
                                    Y = trainingData[:, -1]
                                    X = tf.keras.utils.normalize(X, axis=1)
                                    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=test_size)
                                    ln = 1
                                    neuronsNumber = neurnum
                                    model = tf.keras.Sequential()
                                    while ln <= layerCount:
                                        if (ln == 1):
                                            model.add(tf.keras.layers.Dense(neurnum, activation=act))
                                            layerStr = f"{neurnum}"
                                        else:
                                            neuronsNumber = round(neuronsNumber / 2)
                                            model.add(tf.keras.layers.Dense(neuronsNumber, activation=act))
                                            layerStr += f"/{neuronsNumber}"
                                        ln += 1
                                    print(layerStr)
                                    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
                                    optim = cst.get_optimizer(opt, learning_rate)
                                    model.compile(loss=loss, optimizer=optim, metrics=metrics)
                                    history = model.fit(x = X_train, y = Y_train, epochs = cst.epochs, batch_size=32, validation_data = (X_val, Y_val))
        
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
                                        print("Initial prediction number: " +  str(predictions[i]))
                                        if (predictions[i] < 0.5): predictedClass = 0
                                        if (predictedClass != y_test[i]):
                                            loss_counter += 1
                                            wrong_ids = wrong_ids + "|" + str(test_ids[i])
                                    final_acc =  (1 -  loss_counter / predictions.size)
                                    wrong_ids = wrong_ids[1:]
                                    print(f"Learning Rate: {learning_rate} Test Size: {test_size} Opt: {opt}, Act: {act}, Layers: {layerStr}, Max.Acc: {max(history.history['val_binary_accuracy'])}, Min.Acc: {min(history.history['val_binary_accuracy'])} Max.Loss: {max(history.history['val_loss'])} Min.Loss: {min(history.history['val_loss'])}, Final Acc: {final_acc}, Loss Count: {loss_counter}, Wrong IDS: {wrong_ids} \n")
                                    output = {"optimizer": opt, "activator": act, "layers": layerStr, "maxAcc": max(history.history['val_binary_accuracy']), "minAcc": min(history.history['val_binary_accuracy']), "maxLoss": max(history.history['val_loss']), "minLoss": min(history.history['val_loss']), "jsonModel": model.to_json(), "epochs": cst.epochs, "finalAcc": final_acc, "lossCount": f"{loss_counter}", "testSize": test_size, "learningRate": learning_rate, "wrongIds": wrong_ids}
                                    db.save_model(output)
                                except Exception as e:
                                    print(f"Main exception: {e}")
                                    continue
        print("All models created..")
        db.close_connection()
    
    def analyze_best_models():
        db = DB()
        cst = Customizer()
        plot = Plot()    
        loss = tf.keras.losses.BinaryCrossentropy()
        metrics = [
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
        best_models = db.get_average_best_models()
        for mod in best_models:
            model = None
            trainingData = np.array(db.get_all_table_data("training_data"))
            X = trainingData[:, 1:-1]
            Y = trainingData[:, -1]
            X = tf.keras.utils.normalize(X, axis=1)
            X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=mod[4], shuffle=e)
            model = tf.keras.Sequential()
            splitted_layers = mod[1].split('/')
            for layer in splitted_layers:
                model.add(tf.keras.layers.Dense(layer, activation=mod[2]))
            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
            optim = cst.get_optimizer(mod[3], mod[5])
            model.compile(loss=loss, optimizer=optim, metrics=metrics)
            history = model.fit(x = X_train, y = Y_train, epochs = cst.epochs, validation_data = (X_val, Y_val))
            plot.save_plot(history.history['loss'], history.history['val_loss'], mod[0], 'loss')
            plot.save_plot(history.history['binary_accuracy'], history.history['val_binary_accuracy'], mod[0], 'accuracy')

            testData = np.array(db.get_all_table_data("test_data", True))
            x_test = testData[: , 1:-1]
            test_ids = testData[:, 0]
            y_test = testData[:, -1]
            x_test= tf.keras.utils.normalize(x_test, axis=1)
            predictions = model.predict(x_test)
            loss_counter = 0
            wrong_ids = ""
            for i in range(predictions.size):
                predictedClass = 1
                if predictions[i] < 0.5: predictedClass = 0
                if predictedClass != y_test[i]:
                    loss_counter += 1
                    wrong_ids = wrong_ids + "|" + str(test_ids[i])
            final_acc =  (1 -  loss_counter / predictions.size)
            print(final_acc)