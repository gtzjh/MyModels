import numpy as np
import optuna
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
import yaml, pathlib
from accTest import accTest


###############################################################################
# Decision Tree
def DT(_x_train, _y_train, _cv, _trials, _random_state):
    param_space = {
        "max_depth": lambda t: t.suggest_int("max_depth", 2, 20),
        "min_samples_split": lambda t: t.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": lambda t: t.suggest_int("min_samples_leaf", 1, 20),
        "max_features": lambda t: t.suggest_float("max_features", 0.2, 1),
        "verbose": lambda t: t.suggest_int("verbose", 0, 0),
    }
    return _optimize_model(_x_train, _y_train, _cv, _trials, _random_state, 
                           DecisionTreeRegressor, param_space)
###############################################################################


###############################################################################
# Random Forest
def RF(_x_train, _y_train, _cv, _trials, _random_state):
    param_space = {
        "n_estimators": lambda t: t.suggest_int("n_estimators", 100, 3000, step=100),
        "max_depth": lambda t: t.suggest_int("max_depth", 1, 15),
        "min_samples_leaf": lambda t: t.suggest_int("min_samples_leaf", 1, 5, step=1),
        "min_samples_split": lambda t: t.suggest_int("min_samples_split", 2, 10, step=1),
        "max_features": lambda t: t.suggest_float("max_features", 0.1, 1.0, step=0.1),
        "n_jobs": lambda t: t.suggest_int("n_jobs", -1, -1),
        "verbose": lambda t: t.suggest_int("verbose", 0, 0),
    }
    return _optimize_model(_x_train, _y_train, _cv, _trials, _random_state,
                           RandomForestRegressor, param_space)
###############################################################################


###############################################################################
# GBDT
def GBDT(_x_train, _y_train, _cv, _trials, _random_state):
    param_space = {
        "learning_rate": lambda t: t.suggest_float("learning_rate", 1e-8, 1.0, log=True),
        "n_estimators": lambda t: t.suggest_int("n_estimators", 200, 3000, step=100),
        "max_depth": lambda t: t.suggest_int("max_depth", 1, 15),
        "subsample": lambda t: t.suggest_float("subsample", 0.2, 1.0, step=0.1),
        "min_samples_leaf": lambda t: t.suggest_int("min_samples_leaf", 1, 5, step=1),
        "min_samples_split": lambda t: t.suggest_int("min_samples_split", 2, 10, step=1),
        "max_features": lambda t: t.suggest_float("max_features", 0.2, 1.0, step=0.1),
        "n_jobs": lambda t: t.suggest_int("n_jobs", -1, -1),
        "verbose": lambda t: t.suggest_int("verbose", 0, 0),
    }
    return _optimize_model(_x_train, _y_train, _cv, _trials, _random_state,
                           GradientBoostingRegressor, param_space)
###############################################################################


###############################################################################
# LightGBM
def LGB(_x_train, _y_train, _cv, _trials, _random_state):
    param_space = {
        "max_depth": lambda t: t.suggest_int("max_depth", 1, 15),
        "learning_rate": lambda t: t.suggest_float("learning_rate", 1e-8, 1.0, log=True),
        "n_estimators": lambda t: t.suggest_int("n_estimators", 200, 3000, step=100),
        "num_leaves": lambda t: t.suggest_int("num_leaves", 2, 256),
        "colsample_bytree": lambda t: t.suggest_float("colsample_bytree", 0.2, 1.0),
        "subsample": lambda t: t.suggest_float("subsample", 0.2, 1.0),
        "subsample_freq": lambda t: t.suggest_int("subsample_freq", 1, 7),
        "reg_alpha": lambda t: t.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": lambda t: t.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "n_jobs": lambda t: t.suggest_int("n_jobs", -1, -1),
        "verbose": lambda t: t.suggest_int("verbose", -1, -1),
    }
    return _optimize_model(_x_train, _y_train, _cv, _trials, _random_state,
                           LGBMRegressor, param_space)
###############################################################################


###############################################################################
# Catboost
def CAT(_x_train, _y_train, _cv, _trials, _random_state, _cat_features=None):
    param_space = {
        "iterations": lambda t: t.suggest_int("iterations", 100, 3000, step=100),
        "learning_rate": lambda t: t.suggest_float("learning_rate", 1e-5, 1.0, log=True),
        "max_depth": lambda t: t.suggest_int("max_depth", 3, 10),
        "bagging_temperature": lambda t: t.suggest_float("bagging_temperature", 0.0, 1.0),
        "subsample": lambda t: t.suggest_float("subsample", 0.05, 1.0),
        "colsample_bylevel": lambda t: t.suggest_float("colsample_bylevel", 0.05, 1.0),
        "min_data_in_leaf": lambda t: t.suggest_int("min_data_in_leaf", 1, 100),
        "l2_leaf_reg": lambda t: t.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "leaf_estimation_iterations": lambda t: t.suggest_int("leaf_estimation_iterations", 1, 10),
        # "n_jobs": lambda t: t.suggest_int("n_jobs", -1, -1),
        "verbose": lambda t: t.suggest_int("verbose", 0, 0),
    }
    return _optimize_model(_x_train, _y_train, _cv, _trials, _random_state,
                           lambda **params: CatBoostRegressor(**params, cat_features=_cat_features),
                           param_space)
###############################################################################


###############################################################################
# 新增通用调参函数
def _optimize_model(_x_train, _y_train, _cv, _trials, _random_state, model_class, param_space):
    cv_obj = KFold(n_splits=_cv, shuffle=True, random_state=_random_state)
    
    def _objective(trial):
        params = {
            "random_state": trial.suggest_int("random_state", _random_state, _random_state),
            **{k: v(trial) for k, v in param_space.items()}
        }
        
        cv_r2 = cross_val_score(
            model_class(**params),
            _x_train, _y_train,
            scoring="r2",
            cv=cv_obj,
            n_jobs=-1,
        )
        return np.mean(cv_r2)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(_objective, n_trials=_trials)
    return study
###############################################################################


###############################################################################
# 主函数
def regr(x_train, x_test, y_train, y_test,  # Input train and test data
        model,                              # Model selection
        cv,                                 # Cross-validation for 6 times
        random_state,                       # Global random state setting
        trials,                             # Execute 100 times in optuna
        results_dir,                        # The dir to store the optimization results
        cat_features = None
    ):
    assert model == "cat" or model == "rf" or model == "dt" or model == "lgb" or model == "gbdt"
    assert isinstance(results_dir, pathlib.Path) or isinstance(results_dir, str)
    results_dir = pathlib.Path(results_dir)
    results_dir.mkdir(parents = True, exist_ok = True)


    #######################################################
    # 简化后的模型选择
    MODEL_MAP = {
        "dt": (DT, DecisionTreeRegressor),
        "rf": (RF, RandomForestRegressor),
        "gbdt": (GBDT, GradientBoostingRegressor),
        "lgb": (LGB, LGBMRegressor),
        "cat": (CAT, CatBoostRegressor),
    }

    if model not in MODEL_MAP:
        raise ValueError(f"Invalid model selection: {model}")
    
    study_func, use_model = MODEL_MAP[model]
    study = study_func(
        _x_train=x_train, _y_train=y_train,
        _cv=cv, _trials=trials, _random_state=random_state,
        _cat_features=cat_features
    )
    #######################################################


    ########################################################
    # Return the best trial and parameters
    best_params = study.best_trial.params
    with open(results_dir.joinpath("params.yml"), 'w', encoding = "utf-8") as file:
        yaml.dump(best_params, file)
    print("\n", best_params)
    

    # Train a model base on the optimal parameters
    # On the whole train data.
    best_model = use_model(**best_params)
    best_model.fit(x_train, y_train)


    # Test Accuracy
    y_test_pred = best_model.predict(x_test)
    y_train_pred = best_model.predict(x_train)
    # Output the test and train data, accuracy metrics, and scatter plot for testing.
    accTest(y_test, y_test_pred, y_train, y_train_pred, results_dir)
    ########################################################

    return best_model
###############################################################################