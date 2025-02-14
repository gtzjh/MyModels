import numpy as np
import optuna
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score, KFold
import yaml, pathlib, json
from accTest import accTest
from optuna.samplers import TPESampler


"""
Function calling order:
    -> regr
    -> different kinds of models function:
        (e.g. _DT, _RF, _GBDT, _ADA, _XGB, _LGB, _CAT) 
    -> _optimizer
"""


###############################################################################
# The main function
def regr(
        x_train, x_test, y_train, y_test,
        model,
        cv,
        random_state,
        trials,
        results_dir,
    ):
    """
    Parameters:
        x_train, x_test, y_train, y_test: Input train and test data
        model: Model selection
        cv: Cross-validation for 6 times
        random_state: Global random state setting
        trials: Execute 100 times in optuna
        results_dir: The dir to store the optimization results
    """

    #######################################################
    assert model in ["cat", "rf", "dt", "lgb", "gbdt", "xgb", "ada"], f"Invalid model selection: {model}"
    assert isinstance(results_dir, pathlib.Path) or isinstance(results_dir, str)
    _results_dir = pathlib.Path(results_dir)
    _results_dir.mkdir(parents = True, exist_ok = True)
    #######################################################

    #######################################################
    """
    Global variables declaration:
    random_state, trials, cv, x_train, y_train 作为regr函数的参数传入, 作为全局变量，直接传入 _optimize_model函数
    """
    global global_cv, global_random_state, global_trials, global_x_train, global_y_train
    global_cv = cv
    global_random_state = random_state
    global_trials = trials
    global_x_train = x_train
    global_y_train = y_train
    #######################################################

    #######################################################
    # Model selection
    MODEL_MAP = {
        "dt": (_DT, DecisionTreeRegressor),
        "rf": (_RF, RandomForestRegressor),
        "gbdt": (_GBDT, GradientBoostingRegressor),
        "ada": (_ADA, AdaBoostRegressor),
        "xgb": (_XGB, xgb.XGBRegressor),
        "lgb": (_LGB, LGBMRegressor),
        "cat": (_CAT, CatBoostRegressor),
    }
    
    study_func, use_model = MODEL_MAP[model]   # Select the function and model
    study, static_params = study_func()        # Execute the tuning process
    opt_params = {
        **study.best_trial.params,  # Return the optimal trial and parameters
        **static_params,            # Add static parameters
    }

    opt_model = use_model(**opt_params)        # Train a model base on the optimal parameters
    opt_model.fit(x_train, y_train)            # Train the model
    y_test_pred = opt_model.predict(x_test)    # Predict the test set
    y_train_pred = opt_model.predict(x_train)  # Predict the train set
    accTest(y_test, y_test_pred, y_train, y_train_pred, _results_dir)  # Output the test accuracy
    #######################################################

    #######################################################
    # Save the optimal parameters
    with open(_results_dir.joinpath("params.yml"), 'w', encoding = "utf-8") as file:
        yaml.dump(opt_params, file)
    print("Optimal parameters:\n", json.dumps(opt_params, indent = 4))
    #######################################################

    return opt_model
###############################################################################



###############################################################################
# Decision Tree
def _DT() -> optuna.Study:
    param_space = {
        "max_depth": lambda t: t.suggest_int("max_depth", 2, 20),
        "min_samples_split": lambda t: t.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": lambda t: t.suggest_int("min_samples_leaf", 1, 20),
        "max_features": lambda t: t.suggest_float("max_features", 0.2, 1),
    }
    static_params = {
        "random_state": global_random_state,
    }
    return _optimizer(DecisionTreeRegressor, param_space, static_params)
###############################################################################

###############################################################################
# Random Forest
def _RF() -> optuna.Study:
    param_space = {
        "n_estimators": lambda t: t.suggest_int("n_estimators", 100, 3000, step=100),
        "max_depth": lambda t: t.suggest_int("max_depth", 1, 15),
        "min_samples_leaf": lambda t: t.suggest_int("min_samples_leaf", 1, 5, step=1),
        "min_samples_split": lambda t: t.suggest_int("min_samples_split", 2, 10, step=1),
        "max_features": lambda t: t.suggest_float("max_features", 0.1, 1.0, step=0.1),
    }
    static_params = {
        # "n_jobs": -1,  # Will encounter thread issues in SHAP computation.
        "random_state": global_random_state,
        "verbose": 0,
    }
    return _optimizer(RandomForestRegressor, param_space, static_params)
###############################################################################

###############################################################################
# GBDT
def _GBDT() -> optuna.Study:
    param_space = {
        "learning_rate": lambda t: t.suggest_float("learning_rate", 1e-8, 1.0, log=True),
        "n_estimators": lambda t: t.suggest_int("n_estimators", 100, 3000, step=100),
        "max_depth": lambda t: t.suggest_int("max_depth", 1, 15),
        "subsample": lambda t: t.suggest_float("subsample", 0.2, 1.0, step=0.1),
        "min_samples_leaf": lambda t: t.suggest_int("min_samples_leaf", 1, 5, step=1),
        "min_samples_split": lambda t: t.suggest_int("min_samples_split", 2, 10, step=1),
        "max_features": lambda t: t.suggest_float("max_features", 0.2, 1.0, step=0.1),
    }
    static_params = {
        "random_state": global_random_state,
        "verbose": 0,
    }
    return _optimizer(GradientBoostingRegressor, param_space, static_params)
###############################################################################

###############################################################################
# AdaBoost
def _ADA() -> optuna.Study:
    param_space = {
        "n_estimators": lambda t: t.suggest_int("n_estimators", 100, 3000, step=100),
        "learning_rate": lambda t: t.suggest_float("learning_rate", 0.01, 1.0, log=True),
        "loss": lambda t: t.suggest_categorical("loss", ["linear", "square", "exponential"]),
    }
    static_params = {
        "random_state": global_random_state,
    }
    return _optimizer(AdaBoostRegressor, param_space, static_params)
###############################################################################

###############################################################################
# XGBoost
def _XGB() -> optuna.Study:
    param_space = {
        "max_depth": lambda t: t.suggest_int("max_depth", 1, 15),
        "learning_rate": lambda t: t.suggest_float("learning_rate", 1e-8, 1.0, log=True),
        "n_estimators": lambda t: t.suggest_int("n_estimators", 100, 3000, step=100),
        "subsample": lambda t: t.suggest_float("subsample", 0.2, 1.0, step=0.1),
        "colsample_bytree": lambda t: t.suggest_float("colsample_bytree", 0.2, 1.0, step=0.1),
        "gamma": lambda t: t.suggest_float("gamma", 0, 5, step=0.1),
        "min_child_weight": lambda t: t.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": lambda t: t.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": lambda t: t.suggest_float("reg_lambda", 0.5, 5),
        "tree_method": lambda t: t.suggest_categorical("tree_method", ["hist", "approx"]),
    }   
    static_params = {
        "seed": global_random_state,
        "verbosity": 0,
    }
    return _optimizer(xgb.XGBRegressor, param_space, static_params)
###############################################################################

###############################################################################
# LightGBM
def _LGB() -> optuna.Study:
    param_space = {
        "max_depth": lambda t: t.suggest_int("max_depth", 1, 15),
        "learning_rate": lambda t: t.suggest_float("learning_rate", 1e-8, 1.0, log=True),
        "n_estimators": lambda t: t.suggest_int("n_estimators", 100, 3000, step=100),
        "num_leaves": lambda t: t.suggest_int("num_leaves", 2, 256),
        "colsample_bytree": lambda t: t.suggest_float("colsample_bytree", 0.2, 1.0),
        "subsample": lambda t: t.suggest_float("subsample", 0.2, 1.0),
        "subsample_freq": lambda t: t.suggest_int("subsample_freq", 1, 7),
        "reg_alpha": lambda t: t.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": lambda t: t.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    static_params = {
        "random_state": global_random_state,
        "verbose": -1,
        "n_jobs": -1,
    }
    return _optimizer(LGBMRegressor, param_space, static_params)
###############################################################################

###############################################################################
# Catboost
def _CAT() -> optuna.Study:
    param_space = {
        "iterations": lambda t: t.suggest_int("iterations", 100, 3000, step=100),
        "learning_rate": lambda t: t.suggest_float("learning_rate", 1e-5, 1.0, log=True),
        "max_depth": lambda t: t.suggest_int("max_depth", 3, 10),
        "bagging_temperature": lambda t: t.suggest_float("bagging_temperature", 0.0, 1.0),
        "subsample": lambda t: t.suggest_float("subsample", 0.05, 1.0),
        "colsample_bylevel": lambda t: t.suggest_float("colsample_bylevel", 0.05, 1.0),
        "min_data_in_leaf": lambda t: t.suggest_int("min_data_in_leaf", 1, 100),
        "l2_leaf_reg": lambda t: t.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "leaf_estimation_iterations": lambda t: t.suggest_int("leaf_estimation_iterations", 1, 10)
    }
    static_params = {
        "random_seed": global_random_state,
        "verbose": 0,
    }
    return _optimizer(CatBoostRegressor, param_space, static_params)
###############################################################################


###############################################################################
# Tuning function
def _optimizer(_model, _param_space, _static_params = None) -> tuple[optuna.Study, dict]:
    assert _static_params is None or isinstance(_static_params, dict)
    def _objective(trial) -> float:
        param = {
            **{k: v(trial) for k, v in _param_space.items()},  # Add tuning parameters space
            **_static_params,  # Add static parameters
        }
        cv_r2 = cross_val_score(
            _model(**param),
            global_x_train, global_y_train,
            scoring = "r2",
            # Create a KFold object
            cv = KFold(n_splits = global_cv,
                       random_state = global_random_state,
                       shuffle = True),
            n_jobs = -1,  # Use all available CPU cores for cross validation
        )
        return np.mean(cv_r2)

    """
    Specifying the `sampler` in `optuna.create_study` to be used for the optimization process, 
    and the seed to make the sampler behave in a deterministic way.
    But it will not work when in the parallel mode (`n_jobs > 1` in `study.optimize`).
    https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-obtain-reproducible-optimization-results
    """
    _study = optuna.create_study(
        direction = "maximize",
        sampler = TPESampler(seed = global_random_state),  # Make the sampler behave in a deterministic way.
    )
    _study.optimize(
        _objective,
        n_trials = global_trials,
        n_jobs = 1  # If n_jobs > 1, the optimization process will not be reproducible.
    )

    return _study, _static_params
###############################################################################