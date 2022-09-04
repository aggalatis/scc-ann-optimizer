import tensorflow as tf
import numpy as np
import shap
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt


# Custom Classes
from classes.db import DB
from classes.customizer import Customizer

class FeatureImportanceGenerator:
    def __init__(self):
        self.generate_feature_importance()
    
    def generate_feature_importance(self):
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
            db.close_connection()
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
            shap.initjs()
            explainer = shap.KernelExplainer(model, X_train)
            shap_values = explainer.shap_values(X_val)
            features = [
                'w_quantity',
                'c_quantity',
                'c_spec_surf',
                'f_quantity',
                'f_D90',
                'f_D70',
                'f_D50',
                'f_D30',
                'f_D10',
                'f_spec_grav',
                'f_Sa',
                'fa_quantity',
                'fa_D90',
                'fa_D70',
                'fa_D50',
                'fa_D30',
                'fa_D10',
                'fa_spec_grav',
            ]
            shap.force_plot(explainer.expected_value[0], shap_values[0][0], features=features)
            shap.summary_plot(shap_values[0], X_val, feature_names=features, plot_type="bar")
            
        except Exception as e:
            print(f"Feature Importance Generator exception: {e}")
