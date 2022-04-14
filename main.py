import tensorflow as tf
import numpy as np
import time
import os
from dotenv import load_dotenv

# Custom Classes
from classes.db import DB
from classes.customizer import Customizer



# def main():
#     dataset = np.loadtxt('./data/training.csv', delimiter=';', skiprows=1)
#     x = dataset[:, :-1]
#     y = dataset[:, -1]

#     # Data Normalization
#     myMean =  x.mean(axis = 0)
#     x -= myMean
#     std = x.std(axis = 0)
#     x /= std

#     # x = tf.keras.utils.normalize(x , axis = 1)
#     model = Sequential()
#     model.add(Dense(64, input_dim=(len(x[0, :])), activation='relu'))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dense(16, activation='relu'))
#     model.add(Dense(8, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
#     model.fit(x=x,y=y, epochs= 256, verbose=1)

def main():
    db = DB()
    cst = Customizer()
    trainingData = db.get_all_table_data("training_data")
    validationData = db.get_all_table_data("validation_data")
    
    # Data normalizations
    x = np.array(trainingData)[: , :-1]
    y = np.array(trainingData)[:, -1]
    val_x = np.array(validationData)[: , :-1]
    val_y = np.array(validationData)[:, -1]
    x = tf.keras.utils.normalize(x , axis = 1)

    db.truncate_table("models")
    for opt in cst.optimizers:
        for act in cst.activators:
            for neurnum in cst.neurons:
                for layerCount in cst.hiddenLayers:
                    ln = 1
                    neuronsNumber = neurnum
                    model = tf.keras.Sequential()
                    while ln <= layerCount:
                        if (ln == 1):
                            model.add(tf.keras.layers.Dense(neurnum, input_dim=(len(x[0, :])), activation=act))
                            layerStr = f"{neurnum}"
                        else:
                            neuronsNumber = round(neuronsNumber / 2)
                            model.add(tf.keras.layers.Dense(neuronsNumber, activation=act))
                            layerStr += f"/{neuronsNumber}"
                        ln += 1
                    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
                    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy'])
                    history = model.fit(x=x,y=y,epochs=cst.epochs, validation_data=(val_x, val_y))
                    # output = f"Opt: {opt}, Act: {act}, Layers: {layerStr}, Max.Acc: {max(history.history['val_binary_accuracy'])}, Min.Acc: {min(history.history['val_binary_accuracy'])} Max.Loss: {max(history.history['val_loss'])} Min.Loss: {min(history.history['val_loss'])}\n"
                    output = {"optimizer": opt, "activator": act, "layers": layerStr, "maxAcc": max(history.history['val_binary_accuracy']), "minAcc": min(history.history['val_binary_accuracy']), "maxLoss": max(history.history['val_loss']), "minLoss": min(history.history['val_loss']), "jsonModel": model.to_json(), "epochs": cst.epochs}
                    db.save_model(output)
                    time.sleep(3)
    db.close_connection()

if __name__ == '__main__':
    load_dotenv()
    main()