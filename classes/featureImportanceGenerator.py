import os
import tensorflow as tf
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from catboost import CatBoostClassifier, Pool
from matplotlib import pyplot as plt

shap.initjs()
# Custom Classes
from classes.db import DB
from classes.customizer import Customizer

class FeatureImportanceGenerator:
    def __init__(self):
        self.feature_names = [
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
        shap_values, explainer = self.get_model_shap_values()
        self.generate_bar_plot(shap_values)
        self.generate_decision_plot(shap_values, explainer)
        # self.generate_force_plot(shap_values, explainer)
        self.generate_waterfall_plot(shap_values, explainer)
        # self.export_independed_feature_importance()
    
    def get_model_shap_values(self):
        db = DB()
        trainingData = np.array(db.get_all_table_data("training_data"))
        db.close_connection()
        X = trainingData[:, 1:-1]
        Y = trainingData[:, -1]
        X = tf.keras.utils.normalize(X, axis=1)
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.12)
        layer1 = tf.keras.layers.Input(shape=(18, ))
        layer2 = tf.keras.layers.Dense(72, activation="relu")(layer1)
        layer3 = tf.keras.layers.Dense(36, activation="relu")(layer2)
        layer4 = tf.keras.layers.Dense(18, activation="relu")(layer3)
        output = tf.keras.layers.Dense(1, activation="sigmoid")(layer4)
        func_model = tf.keras.Model(inputs=layer1, outputs=output)

        func_model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adamax(learning_rate=0.004), metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ])
        func_model.fit(x=X_train, y=Y_train, epochs=100, batch_size=32, validation_data = (X_val, Y_val))
        explainer = shap.DeepExplainer(func_model, X_train)
        shap_values = explainer.shap_values(X_val)
        return shap_values, explainer
    
    def generate_bar_plot(self, shap_values):
        shap.summary_plot(shap_values, plot_type = 'bar', feature_names=self.feature_names)
        print("Bar chart exported!")
    
    def generate_decision_plot(self, shap_values, explainer):
        shap.decision_plot(explainer.expected_value[0].numpy(), shap_values[0][0], feature_names = self.feature_names)
        print("Beeswarm chart exported!")

    def generate_force_plot(self, shap_values, explainer):
        shap.force_plot(explainer.expected_value[0].numpy(), shap_values[0][0], feature_names = self.feature_names)
        print("Force chart exported!")

    def generate_waterfall_plot(self, shap_values, explainer):
        shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0].numpy(), shap_values[0][0], feature_names = self.feature_names)
        print("Waterfall chart exported!")
    
    def generate_feature_importance(self, model):

        try:
            explainer = shap.DeepExplainer(model, X_val)
            shap_values = explainer.shap_values(X_val)
            rf_resultX = pd.DataFrame(shap_values[0], columns = self.feature_names)
            vals = np.abs(rf_resultX.values).mean(0)
            shap_importance = pd.DataFrame(list(zip(self.feature_names, vals)),
                                            columns=['col_name','feature_importance_vals'])
            shap_importance.sort_values(by=['feature_importance_vals'],
                                        ascending=False, inplace=True)
            shap.plots.beeswarm(shap_values)
                # shap_importance.head()
                # print(shap_importance) 
                # shap_importance.to_csv(f"D:\\Projects\\sccModel\\featureImportancetests\\test_{i}.csv",sep=';', float_format='{:f}'.format)
                # shap.initjs()
                # print(self.shap_feature_ranking())
                # shap.summary_plot(shap_values[0], X_val, feature_names=self.features, plot_type="bar", title="Feature Importance")   
            
        except Exception as e:
            print(f"Feature Importance Generator exception: {e}")

    def generate_more_plots(self):
        db = DB()
        trainingData = np.array(db.get_all_table_data("training_data"))
        db.close_connection()
        loss = tf.keras.losses.BinaryCrossentropy()
        metrics = [
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]

        try:
            X = trainingData[:, 1:-1]
            Y = trainingData[:, -1]
            X = tf.keras.utils.normalize(X, axis=1)
            X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.12)

            # TODO: Sequential API
            # model = tf.keras.Sequential()
            # model.add(tf.keras.layers.Dense(72, activation="relu"))
            # model.add(tf.keras.layers.Dense(36, activation="relu"))
            # model.add(tf.keras.layers.Dense(18, activation="relu"))
            # model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
            # model.compile(loss=loss, optimizer=tf.keras.optimizers.Adamax(learning_rate=0.004), metrics=metrics)
            # model.fit(x=X_train, y=Y_train, epochs=100, batch_size=32, validation_data = (X_val, Y_val))

            # TODO: Functional API

            layer1 = tf.keras.layers.Input(shape=(18, ))
            layer2 = tf.keras.layers.Dense(72, activation="relu")(layer1)
            layer3 = tf.keras.layers.Dense(36, activation="relu")(layer2)
            layer4 = tf.keras.layers.Dense(18, activation="relu")(layer3)
            output = tf.keras.layers.Dense(1, activation="sigmoid")(layer4)
            func_model = tf.keras.Model(inputs=layer1, outputs=output)
            func_model.summary()

            func_model.compile(loss=loss, optimizer=tf.keras.optimizers.Adamax(learning_rate=0.004), metrics=metrics)
            history = func_model.fit(x=X_train, y=Y_train, epochs=100, batch_size=32, validation_data = (X_val, Y_val))
            print(f"Max.Acc: {max(history.history['val_binary_accuracy'])}, Min.Acc: {min(history.history['val_binary_accuracy'])} Max.Loss: {max(history.history['val_loss'])} Min.Loss: {min(history.history['val_loss'])}")


            explainer = shap.DeepExplainer(func_model, X_train)
            shap_values = explainer.shap_values(X_val)
            # shap.force_plot(base_value=explainer.expected_value[0], shap_values=shap_values[0][0], feature_names=self.feature_names)
            # plt.savefig('force_plot.png')
            # shap.decision_plot(explainer.expected_value[0].numpy(), shap_values[0][0], features = X_val.iloc[0,:], feature_names = self.feature_names)
            # shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0].numpy(), shap_values[0][0], feature_names=self.feature_names, max_display=20)
            # shap.plots.waterfall(shap_values, max_display= 20)
            
        except Exception as e:
            print(f"Feature Importance Generator exception: {e}")

    def export_independed_feature_importance(self):
        db = DB()
        trainingData = np.array(db.get_all_table_data("training_data"))
        db.close_connection()
        X = trainingData[:, 1:-1]
        Y = trainingData[:, -1]
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.15, random_state=1)
        params = {
            'iterations': 622,
            'learning_rate': 0.01,
            'depth': 3,
            'eval_metric': 'AUC',
            'verbose': 200,
            'od_type': 'Iter',
            'od_wait': 500,
            'random_seed': 1
        }

        catModel = CatBoostClassifier(**params)
        catModel.fit(X_train, Y_train, eval_set=(X_val, Y_val), use_best_model=True)
        shap_values = catModel.get_feature_importance(Pool(X_val, label=Y_val), type="ShapValues")
        expected_value= shap_values[0, -1]
        shap_values = shap_values[:, :-1]
        shap.initjs()
        shap.summary_plot(expected_value, X_val[0], feature_names=self.feature_names, plot_type="bar") 

    def shap_feature_ranking(data, shap_values, columns=[]):
        if not columns: columns = data.columns.tolist()     # If columns are not given, take all columns
        
        c_idxs = []
        for column in columns: c_idxs.append(data.columns.get_loc(column))  # Get column locations for desired columns in given dataframe
        if isinstance(shap_values, list):   # If shap values is a list of arrays (i.e., several classes)
            means = [np.abs(shap_values[class_][:, c_idxs]).mean(axis=0) for class_ in range(len(shap_values))]  # Compute mean shap values per class 
            shap_means = np.sum(np.column_stack(means), 1)  # Sum of shap values over all classes 
        else:                               # Else there is only one 2D array of shap values
            assert len(shap_values.shape) == 2, 'Expected two-dimensional shap values array.'
            shap_means = np.abs(shap_values).mean(axis=0)
        
        # Put into dataframe along with columns and sort by shap_means, reset index to get ranking
        df_ranking = pd.DataFrame({'feature': columns, 'mean_shap_value': shap_means}).sort_values(by='mean_shap_value', ascending=False).reset_index(drop=True)
        df_ranking.index += 1
        return df_ranking  
    
