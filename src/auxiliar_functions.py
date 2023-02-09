import shap
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import sys
import warnings
from scipy.stats import percentileofscore
warnings.filterwarnings('ignore') # Pandas warnings

def get_shap_values_one_model(fitted_model, X_train, X_val):
    """
    Get SHAP values as a dataframe
    """
    explainer = shap.TreeExplainer(fitted_model)
    shap_values_val = explainer.shap_values(X_val)

    df_shap_values = pd.DataFrame(shap_values_val, 
                                  columns=X_val.columns,
                                  index=X_val.index).reset_index(drop=True)

    return df_shap_values

def build_model(X_train, y_train, X_val, model_params):
    """
    Fit model
    """
    while True:
        try:
            model = CatBoostRegressor(**model_params)
            categorical_columns = list(X_train.select_dtypes(exclude=["number","bool_","object_"]).columns)
            if len(categorical_columns) == 0:
                model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train, cat_features=categorical_columns)
        except Exception as e:
            print("There was an error with the CatBoost")
            print(e)
            continue
        break
    return model

def introduce_random_variable(X_train, y_train, X_val, model_params):
    """
    Introduce a random variable in a dataframe. 
    The random variable is a permutation of the most influential variable based on SHAP punctuation.
    """
    fitted_model = build_model(X_train, y_train, X_val, model_params)
    df_shap_values = get_shap_values_one_model(fitted_model, X_train, X_val)
    shap_abs_means = df_shap_values.abs().mean()
    most_important_feature = shap_abs_means.sort_values(ascending=False).index[0]

    random_variable_train = X_train[most_important_feature]
    random_variable_val = X_val[most_important_feature]

    sample_train = np.random.uniform(low=random_variable_train.min(), high=random_variable_train.max(), size=(len(X_train),))
    sample_val = np.random.uniform(low=random_variable_val.min(), high=random_variable_val.max(), size=(len(X_val),))

    X_train.loc[:, 'RandomVariable'] = sample_train
    X_val.loc[:, 'RandomVariable'] = sample_val

    return X_train, X_val

def get_shap_values(X_train, y_train, X_val, model_params, n_iterations):
    """
    Get the SHAP puntuaction for each variable n_iterations times.
    It stores the puntuaction, the mean and the max value.
    The seed is different each iteration.
    """
    while True:
        try:
            dict_shap_columns = {c : {'means':[], 'maxs':[]} for c in X_val.columns}
            for iteration in range(n_iterations):
                model_params['random_seed'] = np.random.randint(low=0, high=9999999)
                model = CatBoostRegressor(**model_params) # Restart training
                categorical_columns = list(X_train.select_dtypes(exclude=["number","bool_","object_"]).columns)
                if len(categorical_columns) == 0:
                    model.fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train, cat_features=categorical_columns)
                explainer = shap.TreeExplainer(model)
                shap_values_val = explainer.shap_values(X_val)

                df_abs_shap_values = pd.DataFrame(shap_values_val, 
                                            columns=X_val.columns,
                                            index=X_val.index).abs()
                
                shap_abs_means = df_abs_shap_values.mean()
                shap_abs_maxs = df_abs_shap_values.max()
                
                for column in X_val.columns:
                    dict_shap_columns[column]['means'].append(shap_abs_means[column])
                    dict_shap_columns[column]['maxs'].append(shap_abs_maxs[column])
        except Exception as e:
            print("There was an error with the CatBoost")
            print(e)
            continue
        break
    
    return dict_shap_columns

def remove_first_features_names(dict_shap_columns):
    """
    It selects the features with mean SHAP puntuaction lower than a random variable.
    """
    dict_shap_columns_summary_means = {c: None for c in dict_shap_columns.keys()}
    for feature in dict_shap_columns.keys():
        dict_shap_columns_summary_means[feature] = np.mean(dict_shap_columns[feature]['means'])
    
    mean_random_variable = dict_shap_columns_summary_means['RandomVariable']
    features_to_drop = [feature for feature in list(dict_shap_columns.keys())[:-1] if ((dict_shap_columns_summary_means[feature] <= mean_random_variable))] + ['RandomVariable']

    return features_to_drop, dict_shap_columns_summary_means

def build_residuals_df(X_val, y_val, fitted_model, q_high, q_low):
    """
    It makes the classification of the observations in groups based on the residuals and the quantiles
    """
    preds = fitted_model.predict(X_val).reshape(len(y_val))
    res = (y_val - preds).to_numpy()

    residuals_df = pd.DataFrame({'residuals': res}, index=range(len(res)))
    residuals_df['residual_type'] = 0
    quantile_low = np.quantile(res, q_low)
    quantile_high = np.quantile(res, q_high)
    quantile_star = np.quantile(res, (percentileofscore(res, 0,'weak') / 100))
    if quantile_low < 0 and quantile_high < 0:
        quantile_high = quantile_high - (quantile_low - quantile_star)
        quantile_low = quantile_star
    elif quantile_low > 0 and quantile_high > 0:
        quantile_low = quantile_low - (quantile_high - quantile_star)
        quantile_high = quantile_star
    residuals_df.loc[residuals_df.residuals > quantile_high, "residual_type"] = 1
    residuals_df.loc[residuals_df.residuals < quantile_low, "residual_type"] = -1
    return residuals_df

def give_weight_to_features(residuals_df, df_shap_values):
    """
    It calculates the effect of each variable in each group of observations.
    """
    dict_weights_per_feature = {c:{'count_0': None, 'count_1': None, 'count_-1': None} for c in df_shap_values}
    df_shap_values.loc[:, 'residual_type'] = residuals_df.residual_type

    for column in dict_weights_per_feature.keys():
        df_shap_column = df_shap_values[[column, "residual_type"]]
        df_shap_column.loc[: , 'weight'] = np.sign(df_shap_column[column]) * (df_shap_column[column] ** 2)

        for residual_type in [0, 1, -1]:
            dict_weights_per_feature[column]['count_'+str(residual_type)] = df_shap_column.loc[df_shap_column["residual_type"] == residual_type, 'weight'].sum()
    
    return dict_weights_per_feature

def remove_second_features_names(dict_weights_per_feature, iter, residuals_median):
    """
    It selects the feature with bigger negative influence.
    """
    while True:
        try:
            features_to_drop = []
            dict_features_to_drop = {}
            # possible_features_to_drop = []
            for feature in dict_weights_per_feature.keys():
                if (np.abs(dict_weights_per_feature[feature]['count_0']) + np.abs(dict_weights_per_feature[feature]['count_1'])  + np.abs(dict_weights_per_feature[feature]['count_-1']) < sys.float_info.epsilon):
                    features_to_drop.append(feature)
                elif (residuals_median < 0) and ((dict_weights_per_feature[feature]['count_-1'] >= 0) and (dict_weights_per_feature[feature]['count_1'] >= 0) and (np.abs(dict_weights_per_feature[feature]['count_-1']) > np.abs(dict_weights_per_feature[feature]['count_1']) + np.abs(dict_weights_per_feature[feature]['count_0']))):
                    dict_features_to_drop[feature] = np.abs(dict_weights_per_feature[feature]['count_-1']) - (np.abs(dict_weights_per_feature[feature]['count_1']) + np.abs(dict_weights_per_feature[feature]['count_0'])) 
                elif (residuals_median > 0) and ((dict_weights_per_feature[feature]['count_1'] <= 0) and (dict_weights_per_feature[feature]['count_-1'] <= 0) and (np.abs(dict_weights_per_feature[feature]['count_1']) > np.abs(dict_weights_per_feature[feature]['count_-1']) + np.abs(dict_weights_per_feature[feature]['count_0']))):
                    dict_features_to_drop[feature] = np.abs(dict_weights_per_feature[feature]['count_1']) - (np.abs(dict_weights_per_feature[feature]['count_-1']) + np.abs(dict_weights_per_feature[feature]['count_0']))
                elif ((dict_weights_per_feature[feature]['count_1'] <= 0) and (dict_weights_per_feature[feature]['count_-1'] >= 0) and (np.abs(dict_weights_per_feature[feature]['count_1']) + np.abs(dict_weights_per_feature[feature]['count_-1']) > np.abs(dict_weights_per_feature[feature]['count_0']) )):
                        dict_features_to_drop[feature] = (np.abs(dict_weights_per_feature[feature]['count_1']) + np.abs(dict_weights_per_feature[feature]['count_-1'])) - np.abs(dict_weights_per_feature[feature]['count_0'])    
                # else:
                #     possible_features_to_drop.append(feature)

            # if ((len(dict_features_to_drop.keys()) == 0) and (len(possible_features_to_drop) > 0)):
            #     for feature in possible_features_to_drop:
            #         if ((dict_weights_per_feature[feature]['count_1'] <= 0) and (dict_weights_per_feature[feature]['count_-1'] >= 0) and (np.abs(dict_weights_per_feature[feature]['count_1']) + np.abs(dict_weights_per_feature[feature]['count_-1']) > np.abs(dict_weights_per_feature[feature]['count_0']) )): # + 2 * np.std(dict_weights_per_feature[feature]['count_0'])
            #             dict_features_to_drop[feature] = (np.abs(dict_weights_per_feature[feature]['count_1']) + np.abs(dict_weights_per_feature[feature]['count_-1'])) - np.abs(dict_weights_per_feature[feature]['count_0'])    
                
            if len(features_to_drop) == 0:
                if len(dict_features_to_drop.keys()) > 0:
                    sorted_dict_features_to_drop = {k: v for k, v in sorted(dict_features_to_drop.items(), key=lambda item: item[1], reverse=True)}
                    features_to_drop = list(sorted_dict_features_to_drop.keys())[:1]
        except:
            print("There was an error with the CatBoost")
            continue
        break

    return features_to_drop







