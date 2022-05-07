import tensorflow as tf
import numpy as np
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

# Custom Classes
from classes.db import DB
from classes.customizer import Customizer

def main():
    db = DB()
    cst = Customizer()    
    db.truncate_table("models")
    loss = tf.keras.losses.BinaryCrossentropy()
    metrics = [
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall()
    ]
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
                                model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
                                optim = cst.get_optimizer(opt, learning_rate)
                                model.compile(loss=loss, optimizer=optim, metrics=metrics)
                                history = model.fit(x = X_train, y = Y_train, epochs = cst.epochs, batch_size=32, validation_data = (X_val, Y_val))
                                testData = np.array(db.get_all_table_data("test_data"))
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
                                print(f"Learning Rate: {learning_rate} Test Size: {test_size} Opt: {opt}, Act: {act}, Layers: {layerStr}, Max.Acc: {max(history.history['val_binary_accuracy'])}, Min.Acc: {min(history.history['val_binary_accuracy'])} Max.Loss: {max(history.history['val_loss'])} Min.Loss: {min(history.history['val_loss'])}, Final Acc: {final_acc}, Loss Count: {loss_counter}, Wrong IDS: {wrong_ids} \n")
                                output = {"optimizer": opt, "activator": act, "layers": layerStr, "maxAcc": max(history.history['val_binary_accuracy']), "minAcc": min(history.history['val_binary_accuracy']), "maxLoss": max(history.history['val_loss']), "minLoss": min(history.history['val_loss']), "jsonModel": model.to_json(), "epochs": cst.epochs, "finalAcc": final_acc, "lossCount": f"{loss_counter}", "testSize": test_size, "learningRate": learning_rate, "wrongIds": wrong_ids}
                                db.save_model(output)
                                time.sleep(1)
                            except Exception as e:
                                print(f"Main exception: {e}")
    print("All models ready..")
    db.close_connection()

def publicationModel():
    db = DB()
    A = np.array(db.get_all_table_data("training_data"))
    X = A[:, :-1]
    y = A[:, -1]
    X = tf.keras.utils.normalize(X, axis=1)
    # for c in range(X.shape[1]):
    #     x_col = X[:, c]
    #     mean_value = np.mean(x_col)
    #     std_value = np.std(x_col)
    #     X[:, c] = (x_col - mean_value) / std_value
    
    X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.05)

    model = tf.keras.Sequential(
    [
      tf.keras.layers.Dense(144, activation='sigmoid'),
      tf.keras.layers.Dense(72, activation='sigmoid'),
      tf.keras.layers.Dense(36, activation='sigmoid'),
      tf.keras.layers.Dense(1, activation='sigmoid')
    ]
    )

    opt = tf.keras.optimizers.Adam(
        learning_rate=0.004
    )
    loss = tf.keras.losses.BinaryCrossentropy()
    metrics = [
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall()
    ]   
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    history = model.fit(
        x=X_train,
        y=y_train,
        batch_size=32,
        epochs=100,
        validation_data=(X_val, y_val)
    )

    B = np.array(db.get_all_table_data("test_data"))
    X_test = B[:, :-1]
    Y_test = B[:, -1]
    X_test = tf.keras.utils.normalize(X_test, axis=1)
    # for c in range(X_test.shape[1]):
    #     x_col = X_test[:, c]
    #     mean_value = np.mean(x_col)
    #     std_value = np.std(x_col)
    #     X_test[:, c] = (x_col - mean_value) / std_value
    predictions = model.predict(x=X_test, batch_size=5, verbose=0)
    loss_counter = 0
    for i in range(predictions.size):
        predictedClass = 1
        if (predictions[i] < 0.5): predictedClass = 0
        if (predictedClass != Y_test[i]): loss_counter += 1
    final_acc =  (1 -  loss_counter / predictions.size)
    print("Final acc:", final_acc)




if __name__ == '__main__':
    load_dotenv()
    main()