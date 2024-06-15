import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
import optuna
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score, KFold
from optuna.visualization import plot_optimization_history
from pathlib import Path


###############################################################################
# Neural Network (MLP)
def NN(
        x_train, x_test, y_train, y_test,   # Input train and test data
        cv = 6,                             # Cross-validation for 6 times
        random_state = 42,                  # Random state is 42
        trials = 100,                       # Execute 100 times in optuna
        results_dir = "results/NN"          # The dir to store the optimization results
    ):

    # Check wether the results dir is exist
    results_dir = Path(results_dir)
    if results_dir.exists() == False:
        results_dir.mkdir()

    # 6 Kfold for cross-validation
    cv_obj = KFold(n_splits = cv, shuffle = True, random_state = random_state)
    def objective(trial):
        param = {
            "hidden_layer_sizes": (100, 100, 100),
            "max_iter": 1000,
            "early_stopping": True,
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-5, 1e-1, log=True),
        }
        preditor = MLPRegressor(**param)
        cv_r2 = cross_val_score(
            preditor, 
            x_train, y_train, 
            scoring = "r2",
            cv = cv_obj,
            n_jobs = -1,
            verbose = 0,
        )
        return np.mean(cv_r2)

    # Bayesian
    study = optuna.create_study(direction = "maximize")
    study.optimize(objective, n_trials = trials)

    # 将每次 trial 的验证精度（objective value and best value）保存
    fig_obj = plot_optimization_history(study)
    opt_history_df = pd.DataFrame(
        data = {
            "current_accuracy": np.array(fig_obj.data[0]["y"]),
            "best_accuracy": np.array(fig_obj.data[1]["y"])
        },
        index = pd.Series(data = np.array(fig_obj.data[0]["x"]), name = "trials")
    )
    opt_history_df.to_csv(results_dir.joinpath("optimization.csv"), encoding = "utf-8")

    # 返回最优的trial和参数组合
    trial = study.best_trial
    best_params = trial.params
    best_params.update({
        "hidden_layer_sizes": (100, 100, 100),
        "max_iter": 1000,
        "early_stopping": True,
    })

    # Train a model base upon the optimal parameters on the whole train data.
    best_model = MLPRegressor(**best_params)
    best_model.fit(x_train, y_train)
    y_pred = best_model.predict(x_test)
    accuracy = dict({
        "R2": r2_score(y_test, y_pred),
        "RMSE": root_mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "MAPE": mean_absolute_percentage_error(y_test, y_pred)
    })

    return (best_model, best_params, accuracy)
###############################################################################


###############################################################################
def AG(
        x_train, x_test, y_train, y_test, 
        y_label, 
        train_time = 60*60*2  # About 2 hours
    ):
    train_data = pd.concat([y_train, x_train], axis = 1)
    test_data = pd.concat([y_test, x_test], axis = 1)

    predictor = TabularPredictor(
        label = y_label,
        eval_metric = "r2",
        verbosity = 1
    )
    predictor.fit(
        train_data = train_data,
        time_limit = train_time,
        presets = "best_quality"
    )
    eva = dict(predictor.evaluate(test_data))  # Test the model
    print({
        "R2": round(eva["r2"], 6),
        "MAE": -round(eva["mean_absolute_error"], 6),
        "RMSE": -round(eva["root_mean_squared_error"], 6)
    })

    return None
###############################################################################


###############################################################################
# LightGBM
def LGB(_x_train, _y_train, _cv, _trials, _random_state):
    cv_obj = KFold(n_splits = _cv, shuffle = True, random_state = _random_state)
    def _objective(trial):
        param = {
            "metric": "r2",
            "n_jobs": trial.suggest_int("n_jobs", -1, -1),
            "verbosity": trial.suggest_int("verbosity", -1, -1),
            "random_state": trial.suggest_int("random_state", _random_state, _random_state),
            "max_depth": trial.suggest_int("max_depth", 2, 30),
            "learning_rate": trial.suggest_float("learning_rate", 1e-8, 1.0, log = True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 3000, step = 100),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            "subsample": trial.suggest_float("subsample", 0.2, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log = True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log = True),
        }

        cv_r2 = cross_val_score(
            LGBMRegressor(**param), 
            _x_train, _y_train, 
            scoring = "r2",
            cv = cv_obj,
            n_jobs = -1,
            verbose = 0,
        )
        
        return np.mean(cv_r2)
    
    # Bayesian execution
    _study = optuna.create_study(direction = "maximize")
    _study.optimize(_objective, n_trials = _trials)
    return _study
###############################################################################


###############################################################################
# Decision Tree
def DT(_x_train, _y_train, _cv, _trials, _random_state):
    cv_obj = KFold(n_splits = _cv, shuffle = True, random_state = _random_state)
    def _objective(trial):
        param = {
            "random_state": trial.suggest_int("random_state", _random_state, _random_state),
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_float("max_features", 0.2, 1),
        }

        cv_r2 = cross_val_score(
            DecisionTreeRegressor(**param), 
            _x_train, _y_train, 
            scoring = "r2",
            cv = cv_obj,
            n_jobs = -1,
            verbose = 0,
        )
        return np.mean(cv_r2)

    _study = optuna.create_study(direction = "maximize")
    _study.optimize(_objective, n_trials = _trials)
    return _study
###############################################################################


###############################################################################
# Random Forest
def RF(_x_train, _y_train, _cv, _trials, _random_state):
    cv_obj = KFold(n_splits = _cv, shuffle = True, random_state = _random_state)
    def _objective(trial):
        param = {
            "random_state": trial.suggest_int("random_state", _random_state, _random_state),
            "n_estimators": trial.suggest_int("n_estimators", 100, 3000, step = 100),
            "max_depth": trial.suggest_int("max_depth", 1, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
        }
        cv_r2 = cross_val_score(
            RandomForestRegressor(**param), 
            _x_train, _y_train, 
            scoring = "r2",
            cv = cv_obj,
            n_jobs = -1,
            verbose = 0,
        )
        return np.mean(cv_r2)
    
    _study = optuna.create_study(direction = "maximize")
    _study.optimize(_objective, n_trials = _trials)
    return _study
###############################################################################


###############################################################################
# Catboost
def CAT(_x_train, _y_train, _cv, _trials, _random_state, _cat_features = None):
    cv_obj = KFold(n_splits = _cv, shuffle = True, random_state = _random_state)
    def _objective(trial):
        param = {
            "silent": True,
            "random_state": trial.suggest_int("random_state", _random_state, _random_state),
            "iterations": trial.suggest_int("iterations", 100, 3000, step = 100),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1.0, log = True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "subsample": trial.suggest_float("subsample", 0.05, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log = True),
            "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 1, 10)
        }
        
        cv_r2 = cross_val_score(
            CatBoostRegressor(**param, cat_features = _cat_features), 
            _x_train, _y_train, 
            scoring = "r2",
            cv = cv_obj,
            n_jobs = -1,
            verbose = 0,
        )
        return np.mean(cv_r2)

    _study = optuna.create_study(direction = "maximize")
    _study.optimize(_objective, n_trials = _trials)
    return _study
###############################################################################



###############################################################################
"""
1. Input data.
2. store the optimazation and accuracy results
"""
def ml(
        x_train, x_test, y_train, y_test,  # Input train and test data
        model,                             # Model selection
        cv = 6,                            # Cross-validation for 6 times
        random_state = 6,                  # Global setting
        trials = 100,                      # Execute 100 times in optuna
        results_dir = "results/"           # The dir to store the optimization results
    ):
    assert model == "cat" or model == "rf" or model == "dt" or model == "lgb"
    
    #######################################################
    # Check wether the results dir is exist
    results_dir = Path(results_dir)
    if results_dir.exists() == False:
        results_dir.mkdir()
    #######################################################

    #######################################################
    # Select model
    if model == "cat":
        study = CAT(
            _x_train = x_train, _y_train = y_train,
            _cv = cv,                      # Cross-validation
            _trials = trials,              # How many trials to execute
            _random_state = random_state,  # control the cross-validation split
            _cat_feature = None
        )
        use_model = CatBoostRegressor      # Pass the object, but do not call the method
    elif model == "rf":
        study = RF(
            _x_train = x_train, _y_train = y_train,
            _cv = cv,
            _trials = trials,
            _random_state = random_state
        )
        use_model = RandomForestRegressor
    elif model == "dt":
        study = DT(
            _x_train = x_train, _y_train = y_train,
            _cv = cv,
            _trials = trials,
            _random_state = random_state
        )
        use_model = DecisionTreeRegressor
    elif model == "lgb":
        study = LGB(
            _x_train = x_train, _y_train = y_train,
            _cv = cv,
            _trials = trials,
            _random_state = random_state
        )
        use_model = LGBMRegressor
    else:
        print("Error in model selection.")
        return ValueError
    #######################################################


    ########################################################
    # Return the best trial and parameters
    best_params = study.best_trial.params
    # Train a model base on the optimal parameters
    # On the whole train data.
    best_model = use_model(**best_params)
    best_model.fit(x_train, y_train)
    y_pred = best_model.predict(x_test)
    accuracy = dict({
        "R2": r2_score(y_test, y_pred),
        "RMSE": root_mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "MAPE": mean_absolute_percentage_error(y_test, y_pred)
    })
    ########################################################


    #######################################################
    # Save objective value and best value in every trial
    fig_obj = plot_optimization_history(study)
    opt_history_df = pd.DataFrame(
        data = {
            "current_accuracy": np.array(fig_obj.data[0]["y"]),
            "best_accuracy": np.array(fig_obj.data[1]["y"])
        },
        index = pd.Series(
            data = np.array(fig_obj.data[0]["x"]),
            name = "trials"
        )
    )
    opt_history_df.to_csv(
        results_dir.joinpath("optimization.csv"),
        encoding = "utf-8"
    )
    #######################################################

    # Output the best model, parameters, and accuracy.
    return (best_model, best_params, accuracy)
###############################################################################
