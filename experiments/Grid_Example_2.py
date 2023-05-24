import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt
from BorutaShap import BorutaShap
from powershap import PowerShap
import shapicant
from tqdm import tqdm
from SHAPEffects import FeatureSelector
import random
import shap
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
import sys
from eli5.sklearn import PermutationImportance
from sklearn.feature_selection import SelectFromModel
import pickle


def coeficiente(x, x1, x2, y1, y2):
    return ((y2 - y1) * (x - x1) + (x2 - x1) * y1) / (x2 - x1)


# Taken from PowerSHAP paper
def scores_calc_print(Y, Y_pred, print_bool):
    if len(Y_pred) > 1:
        R2_total = r2_score(Y, Y_pred)
    else:
        R2_total = -1
    RMSE_total = mean_squared_error(Y, Y_pred, squared=False)
    MAE_total = mean_absolute_error(Y, Y_pred)

    if print_bool:
        print(
            tabulate(
                [[RMSE_total, MAE_total, R2_total]],
                ["RMSE", "MAE", "R²"],
                tablefmt="grid",
            )
        )
    else:
        return {"R2": R2_total, "RMSE": RMSE_total, "MAE": MAE_total}


y1s_x2 = [-10, -1, -0.1]
y2s_x2 = [-4, -0.4, -0.04]

y1s_x5 = [10, 1, 0.1]
y2s_x5 = [-25, -2.5, -0.25]

dict_results = {
    "y1_x2": [],
    "y2_x2": [],
    "y1_x5": [],
    "y2_x5": [],
    "media_mae_shapeffects_025_075": [],
    "media_mae_shapeffects_02_08": [],
    "media_mae_shapeffects_015_085": [],
    "media_mae_shapeffects_01_09": [],
    "media_mae_shapeffects_005_095": [],
    "media_mae_powershap": [],
    "media_mae_borutashap": [],
    "media_mae_shapicant": [],
    "media_mae_lasso_001": [],
    "media_mae_lasso_0001": [],
    "media_mae_lasso_00001": [],
    "media_mae_lasso_000001": [],
    "media_mae_boruta": [],
    "media_mae_pimp": [],
    "std_mae_shapeffects_025_075": [],
    "std_mae_shapeffects_02_08": [],
    "std_mae_shapeffects_015_085": [],
    "std_mae_shapeffects_01_09": [],
    "std_mae_shapeffects_005_095": [],
    "std_mae_powershap": [],
    "std_mae_borutashap": [],
    "std_mae_shapicant": [],
    "std_mae_lasso_001": [],
    "std_mae_lasso_0001": [],
    "std_mae_lasso_00001": [],
    "std_mae_lasso_000001": [],
    "std_mae_boruta": [],
    "std_mae_pimp": [],
    "min_mae_shapeffects_025_075": [],
    "min_mae_shapeffects_02_08": [],
    "min_mae_shapeffects_015_085": [],
    "min_mae_shapeffects_01_09": [],
    "min_mae_shapeffects_005_095": [],
    "min_mae_powershap": [],
    "min_mae_borutashap": [],
    "min_mae_shapicant": [],
    "min_mae_lasso_001": [],
    "min_mae_lasso_0001": [],
    "min_mae_lasso_00001": [],
    "min_mae_lasso_000001": [],
    "min_mae_boruta": [],
    "min_mae_pimp": [],
    "max_mae_shapeffects_025_075": [],
    "max_mae_shapeffects_02_08": [],
    "max_mae_shapeffects_015_085": [],
    "max_mae_shapeffects_01_09": [],
    "max_mae_shapeffects_005_095": [],
    "max_mae_powershap": [],
    "max_mae_borutashap": [],
    "max_mae_shapicant": [],
    "max_mae_lasso_001": [],
    "max_mae_lasso_0001": [],
    "max_mae_lasso_00001": [],
    "max_mae_lasso_000001": [],
    "max_mae_boruta": [],
    "max_mae_pimp": [],
    "media_rmse_shapeffects_025_075": [],
    "media_rmse_shapeffects_02_08": [],
    "media_rmse_shapeffects_015_085": [],
    "media_rmse_shapeffects_01_09": [],
    "media_rmse_shapeffects_005_095": [],
    "media_rmse_powershap": [],
    "media_rmse_borutashap": [],
    "media_rmse_shapicant": [],
    "media_rmse_lasso_001": [],
    "media_rmse_lasso_0001": [],
    "media_rmse_lasso_00001": [],
    "media_rmse_lasso_000001": [],
    "media_rmse_boruta": [],
    "media_rmse_pimp": [],
    "std_rmse_shapeffects_025_075": [],
    "std_rmse_shapeffects_02_08": [],
    "std_rmse_shapeffects_015_085": [],
    "std_rmse_shapeffects_01_09": [],
    "std_rmse_shapeffects_005_095": [],
    "std_rmse_powershap": [],
    "std_rmse_borutashap": [],
    "std_rmse_shapicant": [],
    "std_rmse_lasso_001": [],
    "std_rmse_lasso_0001": [],
    "std_rmse_lasso_00001": [],
    "std_rmse_lasso_000001": [],
    "std_rmse_boruta": [],
    "std_rmse_pimp": [],
    "min_rmse_shapeffects_025_075": [],
    "min_rmse_shapeffects_02_08": [],
    "min_rmse_shapeffects_015_085": [],
    "min_rmse_shapeffects_01_09": [],
    "min_rmse_shapeffects_005_095": [],
    "min_rmse_powershap": [],
    "min_rmse_borutashap": [],
    "min_rmse_shapicant": [],
    "min_rmse_lasso_001": [],
    "min_rmse_lasso_0001": [],
    "min_rmse_lasso_00001": [],
    "min_rmse_lasso_000001": [],
    "min_rmse_boruta": [],
    "min_rmse_pimp": [],
    "max_rmse_shapeffects_025_075": [],
    "max_rmse_shapeffects_02_08": [],
    "max_rmse_shapeffects_015_085": [],
    "max_rmse_shapeffects_01_09": [],
    "max_rmse_shapeffects_005_095": [],
    "max_rmse_powershap": [],
    "max_rmse_borutashap": [],
    "max_rmse_shapicant": [],
    "max_rmse_lasso_001": [],
    "max_rmse_lasso_0001": [],
    "max_rmse_lasso_00001": [],
    "max_rmse_lasso_000001": [],
    "max_rmse_boruta": [],
    "max_rmse_pimp": [],
    "media_r2_shapeffects_025_075": [],
    "media_r2_shapeffects_02_08": [],
    "media_r2_shapeffects_015_085": [],
    "media_r2_shapeffects_01_09": [],
    "media_r2_shapeffects_005_095": [],
    "media_r2_powershap": [],
    "media_r2_borutashap": [],
    "media_r2_shapicant": [],
    "media_r2_lasso_001": [],
    "media_r2_lasso_0001": [],
    "media_r2_lasso_00001": [],
    "media_r2_lasso_000001": [],
    "media_r2_boruta": [],
    "media_r2_pimp": [],
    "std_r2_shapeffects_025_075": [],
    "std_r2_shapeffects_02_08": [],
    "std_r2_shapeffects_015_085": [],
    "std_r2_shapeffects_01_09": [],
    "std_r2_shapeffects_005_095": [],
    "std_r2_powershap": [],
    "std_r2_borutashap": [],
    "std_r2_shapicant": [],
    "std_r2_lasso_001": [],
    "std_r2_lasso_0001": [],
    "std_r2_lasso_00001": [],
    "std_r2_lasso_000001": [],
    "std_r2_boruta": [],
    "std_r2_pimp": [],
    "min_r2_shapeffects_025_075": [],
    "min_r2_shapeffects_02_08": [],
    "min_r2_shapeffects_015_085": [],
    "min_r2_shapeffects_01_09": [],
    "min_r2_shapeffects_005_095": [],
    "min_r2_powershap": [],
    "min_r2_borutashap": [],
    "min_r2_shapicant": [],
    "min_r2_lasso_001": [],
    "min_r2_lasso_0001": [],
    "min_r2_lasso_00001": [],
    "min_r2_lasso_000001": [],
    "min_r2_boruta": [],
    "min_r2_pimp": [],
    "max_r2_shapeffects_025_075": [],
    "max_r2_shapeffects_02_08": [],
    "max_r2_shapeffects_015_085": [],
    "max_r2_shapeffects_01_09": [],
    "max_r2_shapeffects_005_095": [],
    "max_r2_powershap": [],
    "max_r2_borutashap": [],
    "max_r2_shapicant": [],
    "max_r2_lasso_001": [],
    "max_r2_lasso_0001": [],
    "max_r2_lasso_00001": [],
    "max_r2_lasso_000001": [],
    "max_r2_boruta": [],
    "max_r2_pimp": [],
}
for y1_x2 in tqdm(y1s_x2):
    for y2_x2 in tqdm(y2s_x2):
        for y1_x5 in tqdm(y1s_x5):
            for y2_x5 in tqdm(y2s_x5):
                os.environ["PYTHONHASHSEED"] = str(1234)
                random.seed(1234)
                np.random.seed(1234)
                dict_results["y1_x2"].append(y1_x2)
                dict_results["y2_x2"].append(y2_x2)
                dict_results["y1_x5"].append(y1_x5)
                dict_results["y2_x5"].append(y2_x5)

                df = pd.DataFrame(np.random.uniform(size=(30001, 10)))
                df.columns = ["X_" + str(number) for number in range(1, 11)]
                df["noise"] = np.random.normal(0, 0.01, size=(30001, 1))
                df["target"] = (
                    2 * df.X_1
                    + y1_x2 * df.X_2**2
                    + 3 * np.sin(2 * np.pi * df.X_3)
                    - 0.4 * df.X_4
                    + y1_x5 * df.X_5**2
                    + (
                        2 * df.X_1.shift(1)
                        + y1_x2 * df.X_2.shift(1) ** 2
                        + 3 * np.sin(2 * np.pi * df.X_3.shift(1))
                        - 0.4 * df.X_4.shift(1)
                        + y1_x5 * df.X_5.shift(1) ** 2
                    )
                    + df.noise
                )
                df.loc[df.index >= 25000, "target"] = (
                    2 * df.loc[df.index >= 25000].X_1
                    + y2_x2 * df.loc[df.index >= 25000].X_2 ** 2
                    + 3 * np.sin(2 * np.pi * df.loc[df.index >= 25000].X_3)
                    - 0.4 * df.loc[df.index >= 25000].X_4
                    + y2_x5 * df.loc[df.index >= 25000].X_5 ** 2
                    + (
                        2 * df.loc[df.index >= 25000].X_1.shift(1)
                        + y2_x2 * df.loc[df.index >= 25000].X_2.shift(1) ** 2
                        + 3 * np.sin(2 * np.pi * df.loc[df.index >= 25000].X_3.shift(1))
                        - 0.4 * df.loc[df.index >= 25000].X_4.shift(1)
                        + y2_x5 * df.loc[df.index >= 25000].X_5.shift(1) ** 2
                    )
                    + df.loc[df.index >= 25000].noise
                )
                   
                df.loc[(df.index > 20000) & (df.index < 25000), "target"] = (
                    2 * df.loc[(df.index > 20000) & (df.index < 25000)].X_1
                    + coeficiente(df.loc[(df.index > 20000) & (df.index < 25000)].index, 20000, 25000, y1_x2, y2_x2) * df.loc[(df.index > 20000) & (df.index < 25000)].X_2 ** 2
                    + 3 * np.sin(2 * np.pi * df.loc[(df.index > 20000) & (df.index < 25000)].X_3)
                    - 0.4 * df.loc[(df.index > 20000) & (df.index < 25000)].X_4
                    + coeficiente(df.loc[(df.index > 20000) & (df.index < 25000)].index, 20000, 25000, y1_x5, y2_x5) * df.loc[(df.index > 20000) & (df.index < 25000)].X_5 ** 2
                    + (
                        2 * df.loc[(df.index > 20000) & (df.index < 25000)].X_1.shift(1)
                        + coeficiente(df.loc[(df.index > 20000) & (df.index < 25000)].index, 20000, 25000, y1_x2, y2_x2) * df.loc[(df.index > 20000) & (df.index < 25000)].X_2.shift(1) ** 2
                        + 3 * np.sin(2 * np.pi * df.loc[(df.index > 20000) & (df.index < 25000)].X_3.shift(1))
                        - 0.4 * df.loc[(df.index > 20000) & (df.index < 25000)].X_4.shift(1)
                        + coeficiente(df.loc[(df.index > 20000) & (df.index < 25000)].index, 20000, 25000, y1_x5, y2_x5) * df.loc[(df.index > 20000) & (df.index < 25000)].X_5.shift(1) ** 2
                    )
                    + df.loc[(df.index > 20000) & (df.index < 25000)].noise  
                   )
                df["target_1"] = df.target.shift(1)

                df = df.dropna().reset_index(drop=True)

                current_db_train = df[df.index < 20000]
                current_db_val = df[df.index.isin(range(20000, 25000))]
                current_db_test = df[df.index >= 25000]

                current_db_train_val = current_db_train.append(current_db_val)

                selected_columns = ["X_" + str(number) for number in range(1, 11)] + [
                    "target_1"
                ]
                target_col = "target"

                X_train = current_db_train[selected_columns]
                y_train = current_db_train[target_col]

                X_val = current_db_val[selected_columns]
                y_val = current_db_val[target_col]

                X_test = current_db_test[selected_columns]
                y_test = current_db_test[target_col]

                X_train_val = X_train.append(X_val)
                y_train_val = y_train.append(y_val)

                seeds = np.random.randint(1, 999999, 50)

                scaler = MinMaxScaler()
                scaler_obj = MinMaxScaler()
                scaler.fit(X_train)
                scaler_obj.fit(y_train.to_numpy().reshape(-1, 1))

                X_train_norm = scaler.transform(X_train)
                X_val_norm = scaler.transform(X_val)
                X_test_norm = scaler.transform(X_test)

                y_train_norm = scaler_obj.transform(
                    y_train.to_numpy().reshape(-1, 1)
                ).reshape(len(y_train))
                y_val_norm = scaler_obj.transform(
                    y_val.to_numpy().reshape(-1, 1)
                ).reshape(len(y_val))
                y_test_norm = scaler_obj.transform(
                    y_test.to_numpy().reshape(-1, 1)
                ).reshape(len(y_test))

                # SHAPEffects 0.25-0.75
                while True:
                    try:
                        model = CatBoostRegressor(
                            verbose=0,
                            n_estimators=250,
                            use_best_model=False,
                            random_seed=123,
                        )
                        feat_sel = FeatureSelector()
                        selected_features_shapeffects_075_025 = feat_sel.fit(
                            X_train, y_train, X_val, y_val, model, 30, 0.25, 0.75
                        )
                        selected_features_shapeffects_075_025 = sorted(
                            list(selected_features_shapeffects_075_025)
                        )

                        dict_results_shapeffects_075_025 = {
                            "rmse": [],
                            "mae": [],
                            "r2": [],
                        }

                        X_test_shapeffects = X_test[
                            selected_features_shapeffects_075_025
                        ]
                        X_train_shapeffects = X_train[
                            selected_features_shapeffects_075_025
                        ]
                        X_val_shapeffects = X_val[selected_features_shapeffects_075_025]
                        for iteration in range(50):
                            print(iteration)
                            model = CatBoostRegressor(
                                verbose=0,
                                n_estimators=250,
                                use_best_model=True,
                                random_seed=seeds[iteration],
                            )
                            model.fit(
                                X_train_shapeffects,
                                y_train,
                                eval_set=(X_val_shapeffects, y_val),
                                verbose=0,
                                plot=False,
                            )

                            test_results = scores_calc_print(
                                y_test,
                                model.predict(X_test_shapeffects),
                                print_bool=False,
                            )

                            dict_results_shapeffects_075_025["r2"].append(
                                test_results["R2"]
                            )
                            dict_results_shapeffects_075_025["rmse"].append(
                                test_results["RMSE"]
                            )
                            dict_results_shapeffects_075_025["mae"].append(
                                test_results["MAE"]
                            )

                        results_shapeffects_075_025_df = pd.DataFrame(
                            dict_results_shapeffects_075_025
                        )
                        mean_shapeffects_075_025_test = (
                            results_shapeffects_075_025_df.mean()
                        )
                        std_shapeffects_075_025_test = (
                            results_shapeffects_075_025_df.std()
                        )
                        max_shapeffects_075_025_test = (
                            results_shapeffects_075_025_df.max()
                        )
                        min_shapeffects_075_025_test = (
                            results_shapeffects_075_025_df.min()
                        )

                        dict_results["media_mae_shapeffects_025_075"].append(
                            mean_shapeffects_075_025_test["mae"]
                        )
                        dict_results["media_rmse_shapeffects_025_075"].append(
                            mean_shapeffects_075_025_test["rmse"]
                        )
                        dict_results["media_r2_shapeffects_025_075"].append(
                            mean_shapeffects_075_025_test["r2"]
                        )
                        dict_results["std_mae_shapeffects_025_075"].append(
                            std_shapeffects_075_025_test["mae"]
                        )
                        dict_results["std_rmse_shapeffects_025_075"].append(
                            std_shapeffects_075_025_test["rmse"]
                        )
                        dict_results["std_r2_shapeffects_025_075"].append(
                            std_shapeffects_075_025_test["r2"]
                        )
                        dict_results["max_mae_shapeffects_025_075"].append(
                            max_shapeffects_075_025_test["mae"]
                        )
                        dict_results["max_rmse_shapeffects_025_075"].append(
                            max_shapeffects_075_025_test["rmse"]
                        )
                        dict_results["max_r2_shapeffects_025_075"].append(
                            max_shapeffects_075_025_test["r2"]
                        )
                        dict_results["min_mae_shapeffects_025_075"].append(
                            min_shapeffects_075_025_test["mae"]
                        )
                        dict_results["min_rmse_shapeffects_025_075"].append(
                            min_shapeffects_075_025_test["rmse"]
                        )
                        dict_results["min_r2_shapeffects_025_075"].append(
                            min_shapeffects_075_025_test["r2"]
                        )
                    except:
                        continue
                    break

                # SHAPEffects 0.2-0.8
                while True:
                    try:
                        model = CatBoostRegressor(
                            verbose=0,
                            n_estimators=250,
                            use_best_model=False,
                            random_seed=123,
                        )
                        feat_sel = FeatureSelector()
                        selected_features_shapeffects_08_02 = feat_sel.fit(
                            X_train, y_train, X_val, y_val, model, 30, 0.2, 0.8
                        )
                        selected_features_shapeffects_08_02 = sorted(
                            list(selected_features_shapeffects_08_02)
                        )

                        dict_results_shapeffects_08_02 = {
                            "rmse": [],
                            "mae": [],
                            "r2": [],
                        }

                        X_test_shapeffects = X_test[selected_features_shapeffects_08_02]
                        X_train_shapeffects = X_train[
                            selected_features_shapeffects_08_02
                        ]
                        X_val_shapeffects = X_val[selected_features_shapeffects_08_02]
                        for iteration in range(50):
                            print(iteration)
                            model = CatBoostRegressor(
                                verbose=0,
                                n_estimators=250,
                                use_best_model=True,
                                random_seed=seeds[iteration],
                            )
                            model.fit(
                                X_train_shapeffects,
                                y_train,
                                eval_set=(X_val_shapeffects, y_val),
                                verbose=0,
                                plot=False,
                            )

                            test_results = scores_calc_print(
                                y_test,
                                model.predict(X_test_shapeffects),
                                print_bool=False,
                            )

                            dict_results_shapeffects_08_02["r2"].append(
                                test_results["R2"]
                            )
                            dict_results_shapeffects_08_02["rmse"].append(
                                test_results["RMSE"]
                            )
                            dict_results_shapeffects_08_02["mae"].append(
                                test_results["MAE"]
                            )

                        results_shapeffects_08_02_df = pd.DataFrame(
                            dict_results_shapeffects_08_02
                        )
                        mean_shapeffects_08_02_test = (
                            results_shapeffects_08_02_df.mean()
                        )
                        std_shapeffects_08_02_test = results_shapeffects_08_02_df.std()
                        max_shapeffects_08_02_test = results_shapeffects_08_02_df.max()
                        min_shapeffects_08_02_test = results_shapeffects_08_02_df.min()

                        dict_results["media_mae_shapeffects_02_08"].append(
                            mean_shapeffects_08_02_test["mae"]
                        )
                        dict_results["media_rmse_shapeffects_02_08"].append(
                            mean_shapeffects_08_02_test["rmse"]
                        )
                        dict_results["media_r2_shapeffects_02_08"].append(
                            mean_shapeffects_08_02_test["r2"]
                        )
                        dict_results["std_mae_shapeffects_02_08"].append(
                            std_shapeffects_08_02_test["mae"]
                        )
                        dict_results["std_rmse_shapeffects_02_08"].append(
                            std_shapeffects_08_02_test["rmse"]
                        )
                        dict_results["std_r2_shapeffects_02_08"].append(
                            std_shapeffects_08_02_test["r2"]
                        )
                        dict_results["max_mae_shapeffects_02_08"].append(
                            max_shapeffects_08_02_test["mae"]
                        )
                        dict_results["max_rmse_shapeffects_02_08"].append(
                            max_shapeffects_08_02_test["rmse"]
                        )
                        dict_results["max_r2_shapeffects_02_08"].append(
                            max_shapeffects_08_02_test["r2"]
                        )
                        dict_results["min_mae_shapeffects_02_08"].append(
                            min_shapeffects_08_02_test["mae"]
                        )
                        dict_results["min_rmse_shapeffects_02_08"].append(
                            min_shapeffects_08_02_test["rmse"]
                        )
                        dict_results["min_r2_shapeffects_02_08"].append(
                            min_shapeffects_08_02_test["r2"]
                        )
                    except:
                        continue
                    break

                # SHAPEffects 0.15-0.85
                while True:
                    try:
                        model = CatBoostRegressor(
                            verbose=0,
                            n_estimators=250,
                            use_best_model=False,
                            random_seed=123,
                        )
                        feat_sel = FeatureSelector()
                        selected_features_shapeffects_085_015 = feat_sel.fit(
                            X_train, y_train, X_val, y_val, model, 30, 0.15, 0.85
                        )
                        selected_features_shapeffects_085_015 = sorted(
                            list(selected_features_shapeffects_085_015)
                        )

                        dict_results_shapeffects_085_015 = {
                            "rmse": [],
                            "mae": [],
                            "r2": [],
                        }

                        X_test_shapeffects = X_test[
                            selected_features_shapeffects_085_015
                        ]
                        X_train_shapeffects = X_train[
                            selected_features_shapeffects_085_015
                        ]
                        X_val_shapeffects = X_val[selected_features_shapeffects_085_015]
                        for iteration in range(50):
                            print(iteration)
                            model = CatBoostRegressor(
                                verbose=0,
                                n_estimators=250,
                                use_best_model=True,
                                random_seed=seeds[iteration],
                            )
                            model.fit(
                                X_train_shapeffects,
                                y_train,
                                eval_set=(X_val_shapeffects, y_val),
                                verbose=0,
                                plot=False,
                            )

                            test_results = scores_calc_print(
                                y_test,
                                model.predict(X_test_shapeffects),
                                print_bool=False,
                            )

                            dict_results_shapeffects_085_015["r2"].append(
                                test_results["R2"]
                            )
                            dict_results_shapeffects_085_015["rmse"].append(
                                test_results["RMSE"]
                            )
                            dict_results_shapeffects_085_015["mae"].append(
                                test_results["MAE"]
                            )

                        results_shapeffects_085_015_df = pd.DataFrame(
                            dict_results_shapeffects_085_015
                        )
                        mean_shapeffects_085_015_test = (
                            results_shapeffects_085_015_df.mean()
                        )
                        std_shapeffects_085_015_test = (
                            results_shapeffects_085_015_df.std()
                        )
                        max_shapeffects_085_015_test = (
                            results_shapeffects_085_015_df.max()
                        )
                        min_shapeffects_085_015_test = (
                            results_shapeffects_085_015_df.min()
                        )

                        dict_results["media_mae_shapeffects_015_085"].append(
                            mean_shapeffects_085_015_test["mae"]
                        )
                        dict_results["media_rmse_shapeffects_015_085"].append(
                            mean_shapeffects_085_015_test["rmse"]
                        )
                        dict_results["media_r2_shapeffects_015_085"].append(
                            mean_shapeffects_085_015_test["r2"]
                        )
                        dict_results["std_mae_shapeffects_015_085"].append(
                            std_shapeffects_085_015_test["mae"]
                        )
                        dict_results["std_rmse_shapeffects_015_085"].append(
                            std_shapeffects_085_015_test["rmse"]
                        )
                        dict_results["std_r2_shapeffects_015_085"].append(
                            std_shapeffects_085_015_test["r2"]
                        )
                        dict_results["max_mae_shapeffects_015_085"].append(
                            max_shapeffects_085_015_test["mae"]
                        )
                        dict_results["max_rmse_shapeffects_015_085"].append(
                            max_shapeffects_085_015_test["rmse"]
                        )
                        dict_results["max_r2_shapeffects_015_085"].append(
                            max_shapeffects_085_015_test["r2"]
                        )
                        dict_results["min_mae_shapeffects_015_085"].append(
                            min_shapeffects_085_015_test["mae"]
                        )
                        dict_results["min_rmse_shapeffects_015_085"].append(
                            min_shapeffects_085_015_test["rmse"]
                        )
                        dict_results["min_r2_shapeffects_015_085"].append(
                            min_shapeffects_085_015_test["r2"]
                        )
                    except:
                        continue
                    break

                # SHAPEffects 0.1-0.9
                while True:
                    try:
                        model = CatBoostRegressor(
                            verbose=0,
                            n_estimators=250,
                            use_best_model=False,
                            random_seed=123,
                        )
                        feat_sel = FeatureSelector()
                        selected_features_shapeffects_09_01 = feat_sel.fit(
                            X_train, y_train, X_val, y_val, model, 30, 0.1, 0.9
                        )
                        selected_features_shapeffects_09_01 = sorted(
                            list(selected_features_shapeffects_09_01)
                        )

                        dict_results_shapeffects_09_01 = {
                            "rmse": [],
                            "mae": [],
                            "r2": [],
                        }

                        X_test_shapeffects = X_test[selected_features_shapeffects_09_01]
                        X_train_shapeffects = X_train[
                            selected_features_shapeffects_09_01
                        ]
                        X_val_shapeffects = X_val[selected_features_shapeffects_09_01]
                        for iteration in range(50):
                            print(iteration)
                            model = CatBoostRegressor(
                                verbose=0,
                                n_estimators=250,
                                use_best_model=True,
                                random_seed=seeds[iteration],
                            )
                            model.fit(
                                X_train_shapeffects,
                                y_train,
                                eval_set=(X_val_shapeffects, y_val),
                                verbose=0,
                                plot=False,
                            )

                            test_results = scores_calc_print(
                                y_test,
                                model.predict(X_test_shapeffects),
                                print_bool=False,
                            )

                            dict_results_shapeffects_09_01["r2"].append(
                                test_results["R2"]
                            )
                            dict_results_shapeffects_09_01["rmse"].append(
                                test_results["RMSE"]
                            )
                            dict_results_shapeffects_09_01["mae"].append(
                                test_results["MAE"]
                            )

                        results_shapeffects_09_01_df = pd.DataFrame(
                            dict_results_shapeffects_09_01
                        )
                        mean_shapeffects_09_01_test = (
                            results_shapeffects_09_01_df.mean()
                        )
                        std_shapeffects_09_01_test = results_shapeffects_09_01_df.std()
                        max_shapeffects_09_01_test = results_shapeffects_09_01_df.max()
                        min_shapeffects_09_01_test = results_shapeffects_09_01_df.min()

                        dict_results["media_mae_shapeffects_01_09"].append(
                            mean_shapeffects_09_01_test["mae"]
                        )
                        dict_results["media_rmse_shapeffects_01_09"].append(
                            mean_shapeffects_09_01_test["rmse"]
                        )
                        dict_results["media_r2_shapeffects_01_09"].append(
                            mean_shapeffects_09_01_test["r2"]
                        )
                        dict_results["std_mae_shapeffects_01_09"].append(
                            std_shapeffects_09_01_test["mae"]
                        )
                        dict_results["std_rmse_shapeffects_01_09"].append(
                            std_shapeffects_09_01_test["rmse"]
                        )
                        dict_results["std_r2_shapeffects_01_09"].append(
                            std_shapeffects_09_01_test["r2"]
                        )
                        dict_results["max_mae_shapeffects_01_09"].append(
                            max_shapeffects_09_01_test["mae"]
                        )
                        dict_results["max_rmse_shapeffects_01_09"].append(
                            max_shapeffects_09_01_test["rmse"]
                        )
                        dict_results["max_r2_shapeffects_01_09"].append(
                            max_shapeffects_09_01_test["r2"]
                        )
                        dict_results["min_mae_shapeffects_01_09"].append(
                            min_shapeffects_09_01_test["mae"]
                        )
                        dict_results["min_rmse_shapeffects_01_09"].append(
                            min_shapeffects_09_01_test["rmse"]
                        )
                        dict_results["min_r2_shapeffects_01_09"].append(
                            min_shapeffects_09_01_test["r2"]
                        )
                    except:
                        continue
                    break

                # SHAPEffects 0.05-0.95
                while True:
                    try:
                        model = CatBoostRegressor(
                            verbose=0,
                            n_estimators=250,
                            use_best_model=False,
                            random_seed=123,
                        )
                        feat_sel = FeatureSelector()
                        selected_features_shapeffects_095_005 = feat_sel.fit(
                            X_train, y_train, X_val, y_val, model, 30, 0.05, 0.95
                        )
                        selected_features_shapeffects_095_005 = sorted(
                            list(selected_features_shapeffects_095_005)
                        )

                        dict_results_shapeffects_095_005 = {
                            "rmse": [],
                            "mae": [],
                            "r2": [],
                        }

                        X_test_shapeffects = X_test[
                            selected_features_shapeffects_095_005
                        ]
                        X_train_shapeffects = X_train[
                            selected_features_shapeffects_095_005
                        ]
                        X_val_shapeffects = X_val[selected_features_shapeffects_095_005]
                        for iteration in range(50):
                            print(iteration)
                            model = CatBoostRegressor(
                                verbose=0,
                                n_estimators=250,
                                use_best_model=True,
                                random_seed=seeds[iteration],
                            )
                            model.fit(
                                X_train_shapeffects,
                                y_train,
                                eval_set=(X_val_shapeffects, y_val),
                                verbose=0,
                                plot=False,
                            )

                            test_results = scores_calc_print(
                                y_test,
                                model.predict(X_test_shapeffects),
                                print_bool=False,
                            )

                            dict_results_shapeffects_095_005["r2"].append(
                                test_results["R2"]
                            )
                            dict_results_shapeffects_095_005["rmse"].append(
                                test_results["RMSE"]
                            )
                            dict_results_shapeffects_095_005["mae"].append(
                                test_results["MAE"]
                            )

                        results_shapeffects_095_005_df = pd.DataFrame(
                            dict_results_shapeffects_095_005
                        )
                        mean_shapeffects_095_005_test = (
                            results_shapeffects_095_005_df.mean()
                        )
                        std_shapeffects_095_005_test = (
                            results_shapeffects_095_005_df.std()
                        )
                        max_shapeffects_095_005_test = (
                            results_shapeffects_095_005_df.max()
                        )
                        min_shapeffects_095_005_test = (
                            results_shapeffects_095_005_df.min()
                        )

                        dict_results["media_mae_shapeffects_005_095"].append(
                            mean_shapeffects_095_005_test["mae"]
                        )
                        dict_results["media_rmse_shapeffects_005_095"].append(
                            mean_shapeffects_095_005_test["rmse"]
                        )
                        dict_results["media_r2_shapeffects_005_095"].append(
                            mean_shapeffects_095_005_test["r2"]
                        )
                        dict_results["std_mae_shapeffects_005_095"].append(
                            std_shapeffects_095_005_test["mae"]
                        )
                        dict_results["std_rmse_shapeffects_005_095"].append(
                            std_shapeffects_095_005_test["rmse"]
                        )
                        dict_results["std_r2_shapeffects_005_095"].append(
                            std_shapeffects_095_005_test["r2"]
                        )
                        dict_results["max_mae_shapeffects_005_095"].append(
                            max_shapeffects_095_005_test["mae"]
                        )
                        dict_results["max_rmse_shapeffects_005_095"].append(
                            max_shapeffects_095_005_test["rmse"]
                        )
                        dict_results["max_r2_shapeffects_005_095"].append(
                            max_shapeffects_095_005_test["r2"]
                        )
                        dict_results["min_mae_shapeffects_005_095"].append(
                            min_shapeffects_095_005_test["mae"]
                        )
                        dict_results["min_rmse_shapeffects_005_095"].append(
                            min_shapeffects_095_005_test["rmse"]
                        )
                        dict_results["min_r2_shapeffects_005_095"].append(
                            min_shapeffects_095_005_test["r2"]
                        )
                    except:
                        continue
                    break

                # Powershap
                while True:
                    try:
                        selector = PowerShap(
                            model=CatBoostRegressor(
                                verbose=0,
                                n_estimators=250,
                                use_best_model=False,
                                random_seed=123,
                            ),
                            power_iterations=10,
                            automatic=True,
                            limit_automatic=10,
                            verbose=True,
                            target_col=target_col,
                        )
                        selector.fit(
                            current_db_train_val[list(selected_columns)],
                            current_db_train_val[target_col],
                        )

                        t = selector._processed_shaps_df
                        selected_features_powershap = t[(t.p_value < 0.01)].index.values

                        selected_features_powershap = sorted(
                            selected_features_powershap
                        )

                        dict_results_powershap = {"rmse": [], "mae": [], "r2": []}

                        X_test_powershap = X_test[selected_features_powershap]
                        X_train_powershap = X_train[selected_features_powershap]
                        X_val_powershap = X_val[selected_features_powershap]
                        for iteration in range(50):
                            print(iteration)
                            model = CatBoostRegressor(
                                verbose=0,
                                n_estimators=250,
                                use_best_model=True,
                                random_seed=seeds[iteration],
                            )
                            model.fit(
                                X_train_powershap,
                                y_train,
                                eval_set=(X_val_powershap, y_val),
                                verbose=0,
                                plot=False,
                            )

                            test_results = scores_calc_print(
                                y_test,
                                model.predict(X_test_powershap),
                                print_bool=False,
                            )

                            dict_results_powershap["r2"].append(test_results["R2"])
                            dict_results_powershap["rmse"].append(test_results["RMSE"])
                            dict_results_powershap["mae"].append(test_results["MAE"])

                        results_powershap_df = pd.DataFrame(dict_results_powershap)
                        mean_powershap_test = results_powershap_df.mean()
                        std_powershap_test = results_powershap_df.std()
                        max_powershap_test = results_powershap_df.max()
                        min_powershap_test = results_powershap_df.min()

                        dict_results["media_mae_powershap"].append(
                            mean_powershap_test["mae"]
                        )
                        dict_results["media_rmse_powershap"].append(
                            mean_powershap_test["rmse"]
                        )
                        dict_results["media_r2_powershap"].append(
                            mean_powershap_test["r2"]
                        )
                        dict_results["std_mae_powershap"].append(
                            std_powershap_test["mae"]
                        )
                        dict_results["std_rmse_powershap"].append(
                            std_powershap_test["rmse"]
                        )
                        dict_results["std_r2_powershap"].append(
                            std_powershap_test["r2"]
                        )
                        dict_results["max_mae_powershap"].append(
                            max_powershap_test["mae"]
                        )
                        dict_results["max_rmse_powershap"].append(
                            max_powershap_test["rmse"]
                        )
                        dict_results["max_r2_powershap"].append(
                            max_powershap_test["r2"]
                        )
                        dict_results["min_mae_powershap"].append(
                            min_powershap_test["mae"]
                        )
                        dict_results["min_rmse_powershap"].append(
                            min_powershap_test["rmse"]
                        )
                        dict_results["min_r2_powershap"].append(
                            min_powershap_test["r2"]
                        )

                    except:
                        continue
                    break

                # BorutaSHAP
                while True:
                    try:
                        model = CatBoostRegressor(
                            verbose=0,
                            n_estimators=250,
                            use_best_model=False,
                            random_seed=123,
                        )

                        # if classification is False it is a Regression problem
                        Feature_Selector = BorutaShap(
                            model=model, importance_measure="shap", classification=False
                        )

                        Feature_Selector.fit(
                            X=X_train_val,
                            y=y_train_val,
                            sample=False,
                            train_or_test="test",
                            normalize=True,
                            verbose=True,
                        )

                        subset = Feature_Selector.Subset()
                        selected_features_borutashap = sorted(subset.columns.values)

                        dict_results_borutashap = {"rmse": [], "mae": [], "r2": []}

                        X_test_borutashap = X_test[selected_features_borutashap]
                        X_train_borutashap = X_train[selected_features_borutashap]
                        X_val_borutashap = X_val[selected_features_borutashap]
                        for iteration in range(50):
                            print(iteration)
                            model = CatBoostRegressor(
                                verbose=0,
                                n_estimators=250,
                                use_best_model=True,
                                random_seed=seeds[iteration],
                            )
                            model.fit(
                                X_train_borutashap,
                                y_train,
                                eval_set=(X_val_borutashap, y_val),
                                verbose=0,
                                plot=False,
                            )

                            test_results = scores_calc_print(
                                y_test,
                                model.predict(X_test_borutashap),
                                print_bool=False,
                            )

                            dict_results_borutashap["r2"].append(test_results["R2"])
                            dict_results_borutashap["rmse"].append(test_results["RMSE"])
                            dict_results_borutashap["mae"].append(test_results["MAE"])

                        results_borutashap_df = pd.DataFrame(dict_results_borutashap)
                        mean_borutashap_test = results_borutashap_df.mean()
                        std_borutashap_test = results_borutashap_df.std()
                        max_borutashap_test = results_borutashap_df.max()
                        min_borutashap_test = results_borutashap_df.min()

                        dict_results["media_mae_borutashap"].append(
                            mean_borutashap_test["mae"]
                        )
                        dict_results["media_rmse_borutashap"].append(
                            mean_borutashap_test["rmse"]
                        )
                        dict_results["media_r2_borutashap"].append(
                            mean_borutashap_test["r2"]
                        )
                        dict_results["std_mae_borutashap"].append(
                            std_borutashap_test["mae"]
                        )
                        dict_results["std_rmse_borutashap"].append(
                            std_borutashap_test["rmse"]
                        )
                        dict_results["std_r2_borutashap"].append(
                            std_borutashap_test["r2"]
                        )
                        dict_results["max_mae_borutashap"].append(
                            max_borutashap_test["mae"]
                        )
                        dict_results["max_rmse_borutashap"].append(
                            max_borutashap_test["rmse"]
                        )
                        dict_results["max_r2_borutashap"].append(
                            max_borutashap_test["r2"]
                        )
                        dict_results["min_mae_borutashap"].append(
                            min_borutashap_test["mae"]
                        )
                        dict_results["min_rmse_borutashap"].append(
                            min_borutashap_test["rmse"]
                        )
                        dict_results["min_r2_borutashap"].append(
                            min_borutashap_test["r2"]
                        )
                    except:
                        continue
                    break

                # Shapicant
                # LightGBM in RandomForest-like mode (with rows subsampling), without columns subsampling
                while True:
                    try:
                        model = CatBoostRegressor(
                            verbose=0,
                            n_estimators=250,
                            use_best_model=False,
                            random_seed=123,
                        )

                        # This is the class (not its instance) of SHAP's TreeExplainer
                        explainer_type = shap.TreeExplainer

                        # Use PandasSelector with 100 iterations
                        selector = shapicant.PandasSelector(
                            model, explainer_type, random_state=42
                        )

                        # Run the feature selection
                        # If we provide a validation set, SHAP values are computed on it, otherwise they are computed on the training set
                        # We can also provide additional parameters to the underlying estimator's fit method through estimator_params
                        selector.fit(
                            X_train, y_train, X_validation=X_val
                        )  # , estimator_params={"categorical_feature": None})

                        # Just get the features list
                        selected_features = selector.get_features()

                        # We can also get the p-values as pandas Series
                        p_values = selector.p_values_

                        selected_features_shapicant = sorted(
                            np.array(selected_features)
                        )

                        dict_results_shapicant = {"rmse": [], "mae": [], "r2": []}

                        X_test_shapicant = X_test[selected_features_shapicant]
                        X_train_shapicant = X_train[selected_features_shapicant]
                        X_val_shapicant = X_val[selected_features_shapicant]
                        for iteration in range(50):
                            print(iteration)
                            model = CatBoostRegressor(
                                verbose=0,
                                n_estimators=250,
                                use_best_model=True,
                                random_seed=seeds[iteration],
                            )
                            model.fit(
                                X_train_shapicant,
                                y_train,
                                eval_set=(X_val_shapicant, y_val),
                                verbose=0,
                                plot=False,
                            )

                            test_results = scores_calc_print(
                                y_test,
                                model.predict(X_test_shapicant),
                                print_bool=False,
                            )

                            dict_results_shapicant["r2"].append(test_results["R2"])
                            dict_results_shapicant["rmse"].append(test_results["RMSE"])
                            dict_results_shapicant["mae"].append(test_results["MAE"])

                        results_shapicant_df = pd.DataFrame(dict_results_shapicant)
                        mean_shapicant_test = results_shapicant_df.mean()
                        std_shapicant_test = results_shapicant_df.std()
                        max_shapicant_test = results_shapicant_df.max()
                        min_shapicant_test = results_shapicant_df.min()

                        dict_results["media_mae_shapicant"].append(
                            mean_shapicant_test["mae"]
                        )
                        dict_results["media_rmse_shapicant"].append(
                            mean_shapicant_test["rmse"]
                        )
                        dict_results["media_r2_shapicant"].append(
                            mean_shapicant_test["r2"]
                        )
                        dict_results["std_mae_shapicant"].append(
                            std_shapicant_test["mae"]
                        )
                        dict_results["std_rmse_shapicant"].append(
                            std_shapicant_test["rmse"]
                        )
                        dict_results["std_r2_shapicant"].append(
                            std_shapicant_test["r2"]
                        )
                        dict_results["max_mae_shapicant"].append(
                            max_shapicant_test["mae"]
                        )
                        dict_results["max_rmse_shapicant"].append(
                            max_shapicant_test["rmse"]
                        )
                        dict_results["max_r2_shapicant"].append(
                            max_shapicant_test["r2"]
                        )
                        dict_results["min_mae_shapicant"].append(
                            min_shapicant_test["mae"]
                        )
                        dict_results["min_rmse_shapicant"].append(
                            min_shapicant_test["rmse"]
                        )
                        dict_results["min_r2_shapicant"].append(
                            min_shapicant_test["r2"]
                        )
                    except:
                        continue
                    break

                # Lasso 0.01
                while True:
                    try:
                        model = Lasso(alpha=0.01)
                        model.fit(X_train_norm, y_train_norm)
                        features_lasso_001 = sorted(
                            list(
                                X_train.columns[
                                    np.abs(model.coef_) > sys.float_info.epsilon
                                ]
                            )
                        )
                        if len(features_lasso_001) > 0:
                            dict_results_lasso_001 = {"rmse": [], "mae": [], "r2": []}

                            X_test_lasso_001 = X_test[features_lasso_001]
                            X_train_lasso_001 = X_train[features_lasso_001]
                            X_val_lasso_001 = X_val[features_lasso_001]
                            for iteration in range(50):
                                print(iteration)
                                model = CatBoostRegressor(
                                    verbose=0,
                                    n_estimators=250,
                                    use_best_model=True,
                                    random_seed=seeds[iteration],
                                )
                                model.fit(
                                    X_train_lasso_001,
                                    y_train,
                                    eval_set=(X_val_lasso_001, y_val),
                                    verbose=0,
                                    plot=False,
                                )

                                test_results = scores_calc_print(
                                    y_test,
                                    model.predict(X_test_lasso_001),
                                    print_bool=False,
                                )

                                dict_results_lasso_001["r2"].append(test_results["R2"])
                                dict_results_lasso_001["rmse"].append(
                                    test_results["RMSE"]
                                )
                                dict_results_lasso_001["mae"].append(
                                    test_results["MAE"]
                                )

                            results_lasso_001_df = pd.DataFrame(dict_results_lasso_001)
                            mean_lasso_001_test = results_lasso_001_df.mean()
                            std_lasso_001_test = results_lasso_001_df.std()
                            max_lasso_001_test = results_lasso_001_df.max()
                            min_lasso_001_test = results_lasso_001_df.min()

                            dict_results["media_mae_lasso_001"].append(
                                mean_lasso_001_test["mae"]
                            )
                            dict_results["media_rmse_lasso_001"].append(
                                mean_lasso_001_test["rmse"]
                            )
                            dict_results["media_r2_lasso_001"].append(
                                mean_lasso_001_test["r2"]
                            )
                            dict_results["std_mae_lasso_001"].append(
                                std_lasso_001_test["mae"]
                            )
                            dict_results["std_rmse_lasso_001"].append(
                                std_lasso_001_test["rmse"]
                            )
                            dict_results["std_r2_lasso_001"].append(
                                std_lasso_001_test["r2"]
                            )
                            dict_results["max_mae_lasso_001"].append(
                                max_lasso_001_test["mae"]
                            )
                            dict_results["max_rmse_lasso_001"].append(
                                max_lasso_001_test["rmse"]
                            )
                            dict_results["max_r2_lasso_001"].append(
                                max_lasso_001_test["r2"]
                            )
                            dict_results["min_mae_lasso_001"].append(
                                min_lasso_001_test["mae"]
                            )
                            dict_results["min_rmse_lasso_001"].append(
                                min_lasso_001_test["rmse"]
                            )
                            dict_results["min_r2_lasso_001"].append(
                                min_lasso_001_test["r2"]
                            )
                        else:
                            dict_results["media_mae_lasso_001"].append(np.nan)
                            dict_results["media_rmse_lasso_001"].append(np.nan)
                            dict_results["media_r2_lasso_001"].append(np.nan)
                            dict_results["std_mae_lasso_001"].append(np.nan)
                            dict_results["std_rmse_lasso_001"].append(np.nan)
                            dict_results["std_r2_lasso_001"].append(np.nan)
                            dict_results["max_mae_lasso_001"].append(np.nan)
                            dict_results["max_rmse_lasso_001"].append(np.nan)
                            dict_results["max_r2_lasso_001"].append(np.nan)
                            dict_results["min_mae_lasso_001"].append(np.nan)
                            dict_results["min_rmse_lasso_001"].append(np.nan)
                            dict_results["min_r2_lasso_001"].append(np.nan)
                    except:
                        continue
                    break

                # Lasso 0.001
                while True:
                    try:
                        model = Lasso(alpha=0.001)
                        model.fit(X_train_norm, y_train_norm)
                        features_lasso_0001 = sorted(
                            list(
                                X_train.columns[
                                    np.abs(model.coef_) > sys.float_info.epsilon
                                ]
                            )
                        )

                        if len(features_lasso_0001) > 0:
                            dict_results_lasso_0001 = {"rmse": [], "mae": [], "r2": []}

                            X_test_lasso_0001 = X_test[features_lasso_0001]
                            X_train_lasso_0001 = X_train[features_lasso_0001]
                            X_val_lasso_0001 = X_val[features_lasso_0001]
                            for iteration in range(50):
                                print(iteration)
                                model = CatBoostRegressor(
                                    verbose=0,
                                    n_estimators=250,
                                    use_best_model=True,
                                    random_seed=seeds[iteration],
                                )
                                model.fit(
                                    X_train_lasso_0001,
                                    y_train,
                                    eval_set=(X_val_lasso_0001, y_val),
                                    verbose=0,
                                    plot=False,
                                )

                                test_results = scores_calc_print(
                                    y_test,
                                    model.predict(X_test_lasso_0001),
                                    print_bool=False,
                                )

                                dict_results_lasso_0001["r2"].append(test_results["R2"])
                                dict_results_lasso_0001["rmse"].append(
                                    test_results["RMSE"]
                                )
                                dict_results_lasso_0001["mae"].append(
                                    test_results["MAE"]
                                )

                            results_lasso_0001_df = pd.DataFrame(
                                dict_results_lasso_0001
                            )
                            mean_lasso_0001_test = results_lasso_0001_df.mean()
                            std_lasso_0001_test = results_lasso_0001_df.std()
                            max_lasso_0001_test = results_lasso_0001_df.max()
                            min_lasso_0001_test = results_lasso_0001_df.min()

                            dict_results["media_mae_lasso_0001"].append(
                                mean_lasso_0001_test["mae"]
                            )
                            dict_results["media_rmse_lasso_0001"].append(
                                mean_lasso_0001_test["rmse"]
                            )
                            dict_results["media_r2_lasso_0001"].append(
                                mean_lasso_0001_test["r2"]
                            )
                            dict_results["std_mae_lasso_0001"].append(
                                std_lasso_0001_test["mae"]
                            )
                            dict_results["std_rmse_lasso_0001"].append(
                                std_lasso_0001_test["rmse"]
                            )
                            dict_results["std_r2_lasso_0001"].append(
                                std_lasso_0001_test["r2"]
                            )
                            dict_results["max_mae_lasso_0001"].append(
                                max_lasso_0001_test["mae"]
                            )
                            dict_results["max_rmse_lasso_0001"].append(
                                max_lasso_0001_test["rmse"]
                            )
                            dict_results["max_r2_lasso_0001"].append(
                                max_lasso_0001_test["r2"]
                            )
                            dict_results["min_mae_lasso_0001"].append(
                                min_lasso_0001_test["mae"]
                            )
                            dict_results["min_rmse_lasso_0001"].append(
                                min_lasso_0001_test["rmse"]
                            )
                            dict_results["min_r2_lasso_0001"].append(
                                min_lasso_0001_test["r2"]
                            )
                        else:
                            dict_results["media_mae_lasso_0001"].append(np.nan)
                            dict_results["media_rmse_lasso_0001"].append(np.nan)
                            dict_results["media_r2_lasso_0001"].append(np.nan)
                            dict_results["std_mae_lasso_0001"].append(np.nan)
                            dict_results["std_rmse_lasso_0001"].append(np.nan)
                            dict_results["std_r2_lasso_0001"].append(np.nan)
                            dict_results["max_mae_lasso_0001"].append(np.nan)
                            dict_results["max_rmse_lasso_0001"].append(np.nan)
                            dict_results["max_r2_lasso_0001"].append(np.nan)
                            dict_results["min_mae_lasso_0001"].append(np.nan)
                            dict_results["min_rmse_lasso_0001"].append(np.nan)
                            dict_results["min_r2_lasso_0001"].append(np.nan)
                    except:
                        continue
                    break

                # Lasso 0.0001
                while True:
                    try:
                        model = Lasso(alpha=0.0001)
                        model.fit(X_train_norm, y_train_norm)
                        features_lasso_00001 = sorted(
                            list(
                                X_train.columns[
                                    np.abs(model.coef_) > sys.float_info.epsilon
                                ]
                            )
                        )
                        dict_results_lasso_00001 = {"rmse": [], "mae": [], "r2": []}

                        X_test_lasso_00001 = X_test[features_lasso_00001]
                        X_train_lasso_00001 = X_train[features_lasso_00001]
                        X_val_lasso_00001 = X_val[features_lasso_00001]
                        for iteration in range(50):
                            print(iteration)
                            model = CatBoostRegressor(
                                verbose=0,
                                n_estimators=250,
                                use_best_model=True,
                                random_seed=seeds[iteration],
                            )
                            model.fit(
                                X_train_lasso_00001,
                                y_train,
                                eval_set=(X_val_lasso_00001, y_val),
                                verbose=0,
                                plot=False,
                            )

                            test_results = scores_calc_print(
                                y_test,
                                model.predict(X_test_lasso_00001),
                                print_bool=False,
                            )

                            dict_results_lasso_00001["r2"].append(test_results["R2"])
                            dict_results_lasso_00001["rmse"].append(
                                test_results["RMSE"]
                            )
                            dict_results_lasso_00001["mae"].append(test_results["MAE"])

                        results_lasso_00001_df = pd.DataFrame(dict_results_lasso_00001)
                        mean_lasso_00001_test = results_lasso_00001_df.mean()
                        std_lasso_00001_test = results_lasso_00001_df.std()
                        max_lasso_00001_test = results_lasso_00001_df.max()
                        min_lasso_00001_test = results_lasso_00001_df.min()

                        dict_results["media_mae_lasso_00001"].append(
                            mean_lasso_00001_test["mae"]
                        )
                        dict_results["media_rmse_lasso_00001"].append(
                            mean_lasso_00001_test["rmse"]
                        )
                        dict_results["media_r2_lasso_00001"].append(
                            mean_lasso_00001_test["r2"]
                        )
                        dict_results["std_mae_lasso_00001"].append(
                            std_lasso_00001_test["mae"]
                        )
                        dict_results["std_rmse_lasso_00001"].append(
                            std_lasso_00001_test["rmse"]
                        )
                        dict_results["std_r2_lasso_00001"].append(
                            std_lasso_00001_test["r2"]
                        )
                        dict_results["max_mae_lasso_00001"].append(
                            max_lasso_00001_test["mae"]
                        )
                        dict_results["max_rmse_lasso_00001"].append(
                            max_lasso_00001_test["rmse"]
                        )
                        dict_results["max_r2_lasso_00001"].append(
                            max_lasso_00001_test["r2"]
                        )
                        dict_results["min_mae_lasso_00001"].append(
                            min_lasso_00001_test["mae"]
                        )
                        dict_results["min_rmse_lasso_00001"].append(
                            min_lasso_00001_test["rmse"]
                        )
                        dict_results["min_r2_lasso_00001"].append(
                            min_lasso_00001_test["r2"]
                        )
                    except:
                        continue
                    break

                # Lasso 0.00001
                while True:
                    try:
                        model = Lasso(alpha=0.00001)
                        model.fit(X_train_norm, y_train_norm)
                        features_lasso_000001 = sorted(
                            list(
                                X_train.columns[
                                    np.abs(model.coef_) > sys.float_info.epsilon
                                ]
                            )
                        )
                        dict_results_lasso_000001 = {"rmse": [], "mae": [], "r2": []}

                        X_test_lasso_000001 = X_test[features_lasso_000001]
                        X_train_lasso_000001 = X_train[features_lasso_000001]
                        X_val_lasso_000001 = X_val[features_lasso_000001]
                        for iteration in range(50):
                            print(iteration)
                            model = CatBoostRegressor(
                                verbose=0,
                                n_estimators=250,
                                use_best_model=True,
                                random_seed=seeds[iteration],
                            )
                            model.fit(
                                X_train_lasso_000001,
                                y_train,
                                eval_set=(X_val_lasso_000001, y_val),
                                verbose=0,
                                plot=False,
                            )

                            test_results = scores_calc_print(
                                y_test,
                                model.predict(X_test_lasso_000001),
                                print_bool=False,
                            )

                            dict_results_lasso_000001["r2"].append(test_results["R2"])
                            dict_results_lasso_000001["rmse"].append(
                                test_results["RMSE"]
                            )
                            dict_results_lasso_000001["mae"].append(test_results["MAE"])

                        results_lasso_000001_df = pd.DataFrame(
                            dict_results_lasso_000001
                        )
                        mean_lasso_000001_test = results_lasso_000001_df.mean()
                        std_lasso_000001_test = results_lasso_000001_df.std()
                        max_lasso_000001_test = results_lasso_000001_df.max()
                        min_lasso_000001_test = results_lasso_000001_df.min()

                        dict_results["media_mae_lasso_000001"].append(
                            mean_lasso_000001_test["mae"]
                        )
                        dict_results["media_rmse_lasso_000001"].append(
                            mean_lasso_000001_test["rmse"]
                        )
                        dict_results["media_r2_lasso_000001"].append(
                            mean_lasso_000001_test["r2"]
                        )
                        dict_results["std_mae_lasso_000001"].append(
                            std_lasso_000001_test["mae"]
                        )
                        dict_results["std_rmse_lasso_000001"].append(
                            std_lasso_000001_test["rmse"]
                        )
                        dict_results["std_r2_lasso_000001"].append(
                            std_lasso_000001_test["r2"]
                        )
                        dict_results["max_mae_lasso_000001"].append(
                            max_lasso_000001_test["mae"]
                        )
                        dict_results["max_rmse_lasso_000001"].append(
                            max_lasso_000001_test["rmse"]
                        )
                        dict_results["max_r2_lasso_000001"].append(
                            max_lasso_000001_test["r2"]
                        )
                        dict_results["min_mae_lasso_000001"].append(
                            min_lasso_000001_test["mae"]
                        )
                        dict_results["min_rmse_lasso_000001"].append(
                            min_lasso_000001_test["rmse"]
                        )
                        dict_results["min_r2_lasso_000001"].append(
                            min_lasso_000001_test["r2"]
                        )
                    except:
                        continue
                    break

                # Boruta
                while True:
                    try:
                        model = CatBoostRegressor(
                            verbose=0,
                            n_estimators=250,
                            use_best_model=False,
                            random_seed=123,
                        )

                        # if classification is False it is a Regression problem
                        Feature_Selector = BorutaShap(
                            model=model, importance_measure="gini", classification=False
                        )

                        Feature_Selector.fit(
                            X=X_train_val,
                            y=y_train_val,
                            sample=False,
                            train_or_test="test",
                            normalize=True,
                            verbose=True,
                        )

                        subset = Feature_Selector.Subset()
                        selected_features_boruta = sorted(subset.columns.values)
                        dict_results_boruta = {"rmse": [], "mae": [], "r2": []}

                        X_test_boruta = X_test[selected_features_boruta]
                        X_train_boruta = X_train[selected_features_boruta]
                        X_val_boruta = X_val[selected_features_boruta]
                        for iteration in range(50):
                            print(iteration)
                            model = CatBoostRegressor(
                                verbose=0,
                                n_estimators=250,
                                use_best_model=True,
                                random_seed=seeds[iteration],
                            )
                            model.fit(
                                X_train_boruta,
                                y_train,
                                eval_set=(X_val_boruta, y_val),
                                verbose=0,
                                plot=False,
                            )

                            test_results = scores_calc_print(
                                y_test, model.predict(X_test_boruta), print_bool=False
                            )

                            dict_results_boruta["r2"].append(test_results["R2"])
                            dict_results_boruta["rmse"].append(test_results["RMSE"])
                            dict_results_boruta["mae"].append(test_results["MAE"])

                        results_boruta_df = pd.DataFrame(dict_results_boruta)
                        mean_boruta_test = results_boruta_df.mean()
                        std_boruta_test = results_boruta_df.std()
                        max_boruta_test = results_boruta_df.max()
                        min_boruta_test = results_boruta_df.min()

                        dict_results["media_mae_boruta"].append(mean_boruta_test["mae"])
                        dict_results["media_rmse_boruta"].append(
                            mean_boruta_test["rmse"]
                        )
                        dict_results["media_r2_boruta"].append(mean_boruta_test["r2"])
                        dict_results["std_mae_boruta"].append(std_boruta_test["mae"])
                        dict_results["std_rmse_boruta"].append(std_boruta_test["rmse"])
                        dict_results["std_r2_boruta"].append(std_boruta_test["r2"])
                        dict_results["max_mae_boruta"].append(max_boruta_test["mae"])
                        dict_results["max_rmse_boruta"].append(max_boruta_test["rmse"])
                        dict_results["max_r2_boruta"].append(max_boruta_test["r2"])
                        dict_results["min_mae_boruta"].append(min_boruta_test["mae"])
                        dict_results["min_rmse_boruta"].append(min_boruta_test["rmse"])
                        dict_results["min_r2_boruta"].append(min_boruta_test["r2"])
                    except:
                        continue
                    break

                # PIMP
                while True:
                    try:
                        model = CatBoostRegressor(
                            verbose=0,
                            n_estimators=250,
                            use_best_model=False,
                            random_seed=123,
                        )
                        model = model.fit(X_train, y_train, verbose=0, plot=False)
                        pimp = PermutationImportance(model, cv="prefit").fit(
                            X_val, y_val
                        )
                        sel = SelectFromModel(pimp, threshold=None, prefit=True)
                        sel.transform(X_train)
                        selected_features_pimp = sorted(
                            list(X_train.columns[sel.get_support()])
                        )
                        dict_results_pimp = {"rmse": [], "mae": [], "r2": []}

                        X_test_pimp = X_test[selected_features_pimp]
                        X_train_pimp = X_train[selected_features_pimp]
                        X_val_pimp = X_val[selected_features_pimp]
                        for iteration in range(50):
                            print(iteration)
                            model = CatBoostRegressor(
                                verbose=0,
                                n_estimators=250,
                                use_best_model=True,
                                random_seed=seeds[iteration],
                            )
                            model.fit(
                                X_train_pimp,
                                y_train,
                                eval_set=(X_val_pimp, y_val),
                                verbose=0,
                                plot=False,
                            )

                            test_results = scores_calc_print(
                                y_test, model.predict(X_test_pimp), print_bool=False
                            )

                            dict_results_pimp["r2"].append(test_results["R2"])
                            dict_results_pimp["rmse"].append(test_results["RMSE"])
                            dict_results_pimp["mae"].append(test_results["MAE"])

                        results_pimp_df = pd.DataFrame(dict_results_pimp)
                        mean_pimp_test = results_pimp_df.mean()
                        std_pimp_test = results_pimp_df.std()
                        max_pimp_test = results_pimp_df.max()
                        min_pimp_test = results_pimp_df.min()

                        dict_results["media_mae_pimp"].append(mean_pimp_test["mae"])
                        dict_results["media_rmse_pimp"].append(mean_pimp_test["rmse"])
                        dict_results["media_r2_pimp"].append(mean_pimp_test["r2"])
                        dict_results["std_mae_pimp"].append(std_pimp_test["mae"])
                        dict_results["std_rmse_pimp"].append(std_pimp_test["rmse"])
                        dict_results["std_r2_pimp"].append(std_pimp_test["r2"])
                        dict_results["max_mae_pimp"].append(max_pimp_test["mae"])
                        dict_results["max_rmse_pimp"].append(max_pimp_test["rmse"])
                        dict_results["max_r2_pimp"].append(max_pimp_test["r2"])
                        dict_results["min_mae_pimp"].append(min_pimp_test["mae"])
                        dict_results["min_rmse_pimp"].append(min_pimp_test["rmse"])
                        dict_results["min_r2_pimp"].append(min_pimp_test["r2"])
                    except:
                        continue
                    break
            with open("dict_results_example_2.pickle", "wb") as handle:
                pickle.dump(dict_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("dict_results_example_2.pickle", "wb") as handle:
            pickle.dump(dict_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("dict_results_example_2.pickle", "wb") as handle:
        pickle.dump(dict_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("dict_results_example_2.pickle", "wb") as handle:
    pickle.dump(dict_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
