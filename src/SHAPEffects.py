"""
Feature selection algorithm based on Shapley values
Author: Carlos Sebastian Martinez-Cava
"""

from auxiliar_functions import introduce_random_variable, get_shap_values, remove_first_features_names, build_model, build_residuals_df, get_shap_values_one_model, give_weight_to_features, remove_second_features_names 
import numpy as np
import copy
from sklearn.metrics import mean_absolute_error
class FeatureSelector:

    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.model = None
        self.X_train_modded = None
        self.X_val_modded = None
        self.model_params = None
        self.n_iterations = None
        self.dict_shap_columns = None
        self.dict_shap_columns_summary_means = None
        self.features_to_drop = []
        self.residuals_df = None
        self.dict_weights_per_feature = None
        self.maes = {}
        self.best_mae = np.inf
        self.best_mae_columns = []

    def fit(self, X_train, y_train, X_val, y_val, model, n_iterations, q_low, q_high):
        """
        X_train: training explanatory variables (pandas dataframe or numpy array)
        y_train: training objective variable (pandas series or numpy array)
        X_val: validation explanatory variables (pandas dataframe or numpy array)
        y_val: validation objective variable (pandas series or numpy array)
        model: fixed model to fit to training data and calculate shapley values on validation data
        n_iterations: number of preprocessing iterations to remove non-relevant variables (int)
        q_low: lower quantile to separate the residuals into classes (float in [0, 1])
        q_high: upper quantile to separate the residuals into classes (float in [0, 1])
        """
        self.X_train = copy.copy(X_train)
        self.y_train = copy.copy(y_train)
        self.X_val = copy.copy(X_val)
        self.y_val = copy.copy(y_val)
        self.model = copy.copy(model)
        self.n_iterations = n_iterations
        self.X_train_modded = self.X_train
        self.X_val_modded = self.X_val

        model_params = model.get_params()
        model_params['verbose'] = 0
        self.model_params = model_params
        if n_iterations > 0:
            print("Preprocessing")
            print("-------------\n")

            print("Introducing a random variable")

            X_train_modded, X_val_modded = introduce_random_variable(self.X_train_modded, 
                                                                     y_train, 
                                                                     self.X_val_modded, 
                                                                     model_params)

            self.X_train_modded = X_train_modded
            self.X_val_modded = X_val_modded
            
            print("Calculating Shapley values")

            dict_shap_columns = get_shap_values(X_train_modded, 
                                                y_train,
                                                X_val_modded,
                                                model_params,
                                                n_iterations)
            
            self.dict_shap_columns = dict_shap_columns

            print("Removing the first features")

            features_to_drop, dict_shap_columns_summary_means = remove_first_features_names(dict_shap_columns)

            self.dict_shap_columns_summary_means = dict_shap_columns_summary_means
            self.features_to_drop = features_to_drop

            self.X_train_modded = self.X_train_modded.drop(features_to_drop, axis = 1)
            self.X_val_modded = self.X_val_modded.drop(features_to_drop, axis = 1)

            print(len(features_to_drop)-1, "features have been removed")
            if len(features_to_drop)-1 > 0:
                print(features_to_drop)
            
            while True:
                try:
                    mae = mean_absolute_error(y_val, 
                                              model.fit(self.X_train_modded, y_train).predict(self.X_val_modded))
                except:
                    print("There was an error with the CatBoost")
                    continue
                break
            self.maes[mae] = self.X_train_modded.columns
            if mae <= self.best_mae:
                self.best_mae = mae
                self.best_mae_columns = self.X_train_modded.columns

        print("\nStarting core module")
        print("--------------------")

        iteration = 0
        features_to_drop_2 = [None]
        while (len(features_to_drop_2) != 0) and (len(self.X_train_modded.columns) > 1):
            print("\nIteration", iteration+1)
            fitted_model = build_model(self.X_train_modded,
                                       self.y_train, 
                                       self.X_val_modded,
                                       self.model_params)

            df_shap_values = get_shap_values_one_model(fitted_model, 
                                                       self.X_train_modded, 
                                                       self.X_val_modded)

            print("\tClasifying residuals")
            
            residuals_df = build_residuals_df(self.X_val_modded,
                                              self.y_val,
                                              fitted_model,
                                              q_high,
                                              q_low)

            self.residuals_df = residuals_df
            residuals_median = residuals_df.residuals.median()

            print("\tGiving weights to features")

            dict_weights_per_feature = give_weight_to_features(residuals_df,
                                                               df_shap_values) 
            
            self.dict_weights_per_feature = dict_weights_per_feature

            print("\tSelecting features")
            
            features_to_drop_2 = remove_second_features_names(self.dict_weights_per_feature, 
                                                              iteration,
                                                              residuals_median)

            self.features_to_drop = np.append(np.array(self.features_to_drop), 
                                              np.array(features_to_drop_2))

            self.X_train_modded = self.X_train_modded.drop(features_to_drop_2, 
                                                           axis = 1)
            self.X_val_modded = self.X_val_modded.drop(features_to_drop_2, 
                                                       axis = 1)

            print("\t" + str(len(features_to_drop_2)), "features have been removed")
            if len(features_to_drop_2) > 0:
                print("\t", features_to_drop_2)
   
            print("\t" + str(self.X_train_modded.shape[1]), "features left")
            iteration += 1
            while True:
                try:
                    mae = mean_absolute_error(y_val, 
                                              model.fit(self.X_train_modded, y_train).predict(self.X_val_modded))
                except Exception as e:
                    print("There was an error with the CatBoost")
                    print(e)
                    continue
                break
            self.maes[mae] = self.X_train_modded.columns
            if mae <= self.best_mae:
                self.best_mae = mae
                self.best_mae_columns = self.X_train_modded.columns

        print(len(self.X_train_modded.columns), "features have been selected out of", len(self.X_train.columns))

        return self.best_mae_columns





        
