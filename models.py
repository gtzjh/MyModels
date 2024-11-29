import numpy as np
import pandas as pd
import optuna
from optuna.visualization import plot_optimization_history
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score, KFold
from pathlib import Path
import os, yaml


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
            "max_depth": trial.suggest_int("max_depth", 1, 15),
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
# Catboost
def CAT(_x_train, _y_train, _cv, _trials, _random_state, _cat_features = None):
    cv_obj = KFold(n_splits = _cv, shuffle = True, random_state = _random_state)
    def _objective(trial):
        param = {
            "verbose": trial.suggest_int("verbose", 0, 0),
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
# Random Forest
def RF(_x_train, _y_train, _cv, _trials, _random_state):
    cv_obj = KFold(n_splits = _cv, shuffle = True, random_state = _random_state)
    def _objective(trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 3000, step = 100),
            "max_depth": trial.suggest_int("max_depth", 1, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5, step = 1),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10, step = 1),
            "max_features": trial.suggest_float("max_features", 0.1, 1.0, step = 0.1),
            "random_state": trial.suggest_int("random_state", _random_state, _random_state),
            "n_jobs": trial.suggest_int("n_jobs", -1, -1)
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
# GBDT
def GBDT(_x_train, _y_train, _cv, _trials, _random_state):
    cv_obj = KFold(n_splits = _cv, shuffle = True, random_state = _random_state)
    def _objective(trial):
        param = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-8, 1.0, log = True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 3000, step = 100),
            "max_depth": trial.suggest_int("max_depth", 1, 15),
            "subsample": trial.suggest_float("subsample", 0.2, 1.0, step = 0.1),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5, step = 1),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10, step = 1),
            "max_features": trial.suggest_float("max_features", 0.2, 1.0, step = 0.1),
            "random_state": trial.suggest_int("random_state", _random_state, _random_state),
        }

        cv_r2 = cross_val_score(
            GradientBoostingRegressor(**param), 
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



def ml(
        x_train, x_test, y_train, y_test,  # Input train and test data
        model,                             # Model selection
        cv = 6,                            # Cross-validation for 6 times
        random_state = 6,                  # Global setting
        trials = 100,                      # Execute 100 times in optuna
        results_dir = "results/",          # The dir to store the optimization results
        cat_features = None
    ):
    assert model == "cat" or model == "rf" or model == "dt" or model == "lgb" or model == "gbdt"
    
    #######################################################
    # Check wether the results dir is exist
    results_dir = Path(results_dir)
    if results_dir.exists() == False:
        os.makedirs(str(results_dir))
    #######################################################

    #######################################################
    # Select model
    if model == "cat":
        study = CAT(
            _x_train = x_train, _y_train = y_train,
            _cv = cv,                      # Cross-validation
            _trials = trials,              # How many trials to execute
            _random_state = random_state,  # control the cross-validation split
            _cat_features = cat_features   # Categories features
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
    elif model == "gbdt":
        study = GBDT(
            _x_train = x_train, _y_train = y_train,
            _cv = cv,
            _trials = trials,
            _random_state = random_state
        )
        use_model = GradientBoostingRegressor
    else:
        print("Error in model selection.")
        return ValueError
    #######################################################


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


    ########################################################
    # Return the best trial and parameters
    best_params = study.best_trial.params
    print(best_params)
    with open(results_dir.joinpath("params.yml"), 'w', encoding = "utf-8") as file:
        yaml.dump(best_params, file)
    

    # Train a model base on the optimal parameters
    # On the whole train data.
    best_model = use_model(**best_params)
    best_model.fit(x_train, y_train)

    #-------------------------------------------------------
    # Accuracy on test set
    y_pred = best_model.predict(x_test)
    test_accuracy = dict({
        "R2": float(r2_score(y_test, y_pred)),
        "RMSE": float(root_mean_squared_error(y_test, y_pred)),
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_test, y_pred))
    })
    scatter_test = pd.DataFrame(data = {
        "y_test": y_test,
        "y_pred": y_pred
    })
    scatter_test.to_csv(
        results_dir.joinpath("scatter_test.csv"),
        encoding = "utf-8", index = False
    )
    print(test_accuracy)
    #-------------------------------------------------------

    #-------------------------------------------------------
    # Accuracy on training set
    y_train_pred = best_model.predict(x_train)
    train_accuracy = dict({
        "R2": float(r2_score(y_train, y_train_pred)),
        "RMSE": float(root_mean_squared_error(y_train, y_train_pred)),
        "MAE": float(mean_absolute_error(y_train, y_train_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_train, y_train_pred))
    })
    scatter_train = pd.DataFrame(data = {
        "y_train": y_train,
        "y_train_pred": y_train_pred
    })
    scatter_train.to_csv(
        results_dir.joinpath("scatter_train.csv"),
        encoding = "utf-8", index = False
    )
    print(train_accuracy)
    #-------------------------------------------------------

    accuracy_dict = dict({
        "test_accuracy": test_accuracy,
        "train_accuracy": train_accuracy
    })
    with open(results_dir.joinpath("accuracy.yml"), 'w', encoding = "utf-8") as file:
        yaml.dump(accuracy_dict, file)
    ########################################################

    return best_model
