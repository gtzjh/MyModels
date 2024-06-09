import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score, KFold
from optuna.visualization import plot_optimization_history
from pathlib import Path


###############################################################################
# LightGBM
###############################################################################


###############################################################################
# Decision Tree
def DT(_x_train, _y_train, _cv, _trials, _random_state):
    cv_obj = KFold(n_splits = _cv, shuffle = True, random_state = _random_state)
    def objective(trial):
        param = {
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

    # Bayesian execution
    _study = optuna.create_study(direction = "maximize")
    _study.optimize(objective, n_trials = _trials)
    return _study
###############################################################################


###############################################################################
# Random Forest
###############################################################################


###############################################################################
# Catboost
def CAT(_x_train, _y_train, _cv, _trials, _random_state):
    cv_obj = KFold(n_splits = _cv, shuffle = True, random_state = _random_state)
    def objective(trial):
        param = {
            "silent": True,
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
            CatBoostRegressor(**param), 
            _x_train, _y_train, 
            scoring = "r2",
            cv = cv_obj,
            n_jobs = -1,
            verbose = 0,
        )
        return np.mean(cv_r2)

    # Bayesian execution
    _study = optuna.create_study(direction = "maximize")
    _study.optimize(objective, n_trials = _trials)
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
        random_state = 42,                 # Random state is 42 Global
        trials = 100,                      # Execute 100 times in optuna
        results_dir = "results/"           # The dir to store the optimization results
    ):
    assert model == "cat" or model == "rf" or model == "dt"
    
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
            _random_state = random_state   # control the cross-validation split
        )
        final_model = CatBoostRegressor    # Pass the object, but do not call the method
    elif model == "rf":
        return None
    elif model == "dt":
        study = DT(
            _x_train = x_train, _y_train = y_train,
            _cv = cv,
            _trials = trials,
            _random_state = random_state
        )
        final_model = DecisionTreeRegressor
    else:
        return ValueError
    #######################################################


    ########################################################
    # Return the best trial and parameters
    best_params = study.best_trial.params
    # Train a model base on the optimal parameters
    # On the whole train data.
    best_model = final_model(**best_params)
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
