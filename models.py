import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score, KFold
import yaml, pathlib
from accuracy import RegrAccuracy



"""A class for training and optimizing various regression models."""
class Regr:
    def __init__(self, cv: int, random_state: int, trials: int, results_dir: str | pathlib.Path):
        """
        Initialize the Regr class.
        Parameters:
            cv (int): Number of folds for cross-validation
            random_state (int): Random seed for reproducibility
            trials (int): Number of trials to execute in optuna optimization
            results_dir (str or pathlib.Path): Directory path to store the optimization results
        """
        
        self.cv = cv
        self.random_state = random_state
        self.trials = trials
        self.results_dir = pathlib.Path(results_dir)
        self.results_dir.mkdir(parents = True, exist_ok = True)
        
        # Model mapping dictionary
        self.MODEL_MAP = {
            "dt": (self._DT, DecisionTreeRegressor),
            "rf": (self._RF, RandomForestRegressor),
            "gbdt": (self._GBDT, GradientBoostingRegressor),
            "ada": (self._ADA, AdaBoostRegressor),
            "xgb": (self._XGB, xgb.XGBRegressor),
            "lgb": (self._LGB, LGBMRegressor),
            "cat": (self._CAT, CatBoostRegressor),
            "svr": (self._SVR, SVR),
            "knr": (self._KNR, KNeighborsRegressor),
            "mlp": (self._MLP, MLPRegressor),
        }

    ###########################################################################
    def fit(self, x_train, y_train, model: str, cat_features = None | list[str]) -> any:
        """
        Train and optimize a regression model.
        Parameters:
            x_train (pd.DataFrame): Training features data
            y_train (pd.Series): Training target data
            model (str): 
                Model selection,
                must beone of ["cat", "rf", "dt", "lgb", "gbdt", "xgb", "ada", "svr", "knr", "mlp"]
            cat_features (list[str] or None): 
                List of categorical feature names, if any
        Returns:
            opt_model (any): The optimized model
        """
        assert model in self.MODEL_MAP, \
            f"Invalid model selection: {model}, it must be one of {list(self.MODEL_MAP.keys())}"
        
        # Add data validation
        if len(x_train) != len(y_train):
            raise ValueError("x_train and y_train must have the same length")
        if cat_features is not None and not all(feat in x_train.columns for feat in cat_features):
            raise ValueError("All categorical features must exist in x_train")
        
        self.x_train = x_train
        self.y_train = y_train
        self.cat_features = cat_features
        
        study_func, use_model = self.MODEL_MAP[model]
        study, static_params = study_func()
        opt_params = {**static_params, **study.best_trial.params}
        
        # Train model with optimal parameters
        opt_model = use_model(**opt_params)
        opt_model.fit(x_train, y_train)

        # Save optimal parameters
        with open(self.results_dir.joinpath("params.yml"), 'w', encoding = "utf-8") as file:
            yaml.dump(opt_params, file)
        # print("Optimal parameters:\n", json.dumps(opt_params, indent = 4))

        # Output the optimized model
        return opt_model
    ###########################################################################

    ###########################################################################
    def evaluate(self, opt_model, x_test, y_test, x_train, y_train) -> None:
        """
        Evaluate the optimized model and save the results.
        Parameters:
            opt_model: The optimized model
            x_test: The testing features data
            y_test: The testing target data
            x_train: The training features data
            y_train: The training target data
        """
        # Evaluate and save results
        y_test_pred = opt_model.predict(x_test)    # Accuracy on testing set
        y_train_pred = opt_model.predict(x_train)  # Accuracy on training set
        RegrAccuracy(y_test, y_test_pred, y_train, y_train_pred, self.results_dir)
        
        return None
    ###########################################################################

    ###########################################################################
    def _optimizer(self, _model, _param_space, _static_params=None) -> tuple[optuna.Study, None|dict]:
        """
        Internal method for model optimization using optuna.
        Parameters:
            _model: The model to optimize
            _param_space: The parameter space to optimize
            _static_params: Static parameters for the model
        """
        if _static_params is None:
            _static_params = {}
        assert isinstance(_static_params, dict)
        
        def _objective(trial):
            param = {
                **{k: v(trial) for k, v in _param_space.items()},
                **_static_params,
            }
            cv_r2 = cross_val_score(
                _model(**param),
                self.x_train, self.y_train,
                scoring="r2",
                cv=KFold(n_splits=self.cv, random_state=self.random_state, shuffle=True),
                n_jobs=-1,
            )
            return np.mean(cv_r2)

        _study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.random_state),
        )
        _study.optimize(
            _objective,
            n_trials=self.trials,
            n_jobs=1
        )
        return _study, _static_params
    ###########################################################################


    ###########################################################################
    """
    Support Vector Regression regressor
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
    """
    def _SVR(self) -> tuple[optuna.Study, dict]:
        param_space = {
            "kernel": lambda t: t.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"]),
            "C": lambda t: t.suggest_float("C", 0.01, 100, log=True),
            "epsilon": lambda t: t.suggest_float("epsilon", 0.01, 1.0, step=0.01),
        }
        static_params = {
            "verbose": 0,
        }
        return self._optimizer(SVR, param_space, static_params)
    ###########################################################################

    ###########################################################################
    """
    K-Nearest Neighbors regressor
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
    """
    def _KNR(self) -> tuple[optuna.Study, dict]:
        param_space = {
            "n_neighbors": lambda t: t.suggest_int("n_neighbors", 1, 100, step=1),
            "weights": lambda t: t.suggest_categorical("weights", ["uniform", "distance"]),
            "leaf_size": lambda t: t.suggest_int("leaf_size", 1, 100, step=1)
        }
        static_params = {
            "n_jobs": -1,
        }
        return self._optimizer(KNeighborsRegressor, param_space, static_params)
    ###########################################################################

    ###########################################################################
    """
    Multi-Layer Perceptron regressor
    https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
    """
    def _MLP(self) -> tuple[optuna.Study, dict]:
        param_space = {
            "alpha": lambda t: t.suggest_float("alpha", 0.0001, 0.1, log=True),
            "learning_rate_init": lambda t: t.suggest_float("learning_rate_init", 0.0001, 0.1, log=True),
            "max_iter": lambda t: t.suggest_int("max_iter", 100, 3000, step = 100),
        }
        static_params = {
            "hidden_layer_sizes": (200, 200, 200),
            "activation": "relu",
            "solver": "adam",
            "batch_size": "auto",
            "random_state": self.random_state,
            "verbose": 0
        }
        return self._optimizer(MLPRegressor, param_space, static_params)
    ###########################################################################

    ###########################################################################
    """
    Decision Tree regressor
    https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
    """
    def _DT(self) -> tuple[optuna.Study, dict]:
        param_space = {
            "max_depth": lambda t: t.suggest_int("max_depth", 2, 20),
            "min_samples_split": lambda t: t.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": lambda t: t.suggest_int("min_samples_leaf", 1, 20),
            "max_features": lambda t: t.suggest_float("max_features", 0.2, 1),
        }
        static_params = {"random_state": self.random_state}
        return self._optimizer(DecisionTreeRegressor, param_space, static_params)
    ###########################################################################

    ###########################################################################
    """
    Random Forest regressor
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    """
    def _RF(self) -> tuple[optuna.Study, dict]:
        param_space = {
            "n_estimators": lambda t: t.suggest_int("n_estimators", 100, 3000, step=100),
            "max_depth": lambda t: t.suggest_int("max_depth", 1, 15),
            "min_samples_leaf": lambda t: t.suggest_int("min_samples_leaf", 1, 5, step=1),
            "min_samples_split": lambda t: t.suggest_int("min_samples_split", 2, 10, step=1),
            "max_features": lambda t: t.suggest_float("max_features", 0.1, 1.0, step=0.1),
        }
        static_params = {
            "random_state": self.random_state,
            "verbose": 0,
        }
        return self._optimizer(RandomForestRegressor, param_space, static_params)
    ###########################################################################

    ###########################################################################
    """
    Gradient Boosting regressor
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
    """
    def _GBDT(self) -> tuple[optuna.Study, dict]:
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
            "random_state": self.random_state,
            "verbose": 0,
        }
        return self._optimizer(GradientBoostingRegressor, param_space, static_params)
    ###########################################################################

    ###########################################################################
    """
    AdaBoost regressor
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
    """
    def _ADA(self) -> tuple[optuna.Study, dict]:
        param_space = {
            "n_estimators": lambda t: t.suggest_int("n_estimators", 100, 3000, step=100),
            "learning_rate": lambda t: t.suggest_float("learning_rate", 0.01, 1.0, log=True),
            "loss": lambda t: t.suggest_categorical("loss", ["linear", "square", "exponential"]),
        }
        static_params = {
            "random_state": self.random_state,
        }
        return self._optimizer(AdaBoostRegressor, param_space, static_params)
    ###########################################################################

    ###########################################################################
    """
    XGBoost regressor
    https://xgboost.readthedocs.io/en/latest/python/python_api.html
    """
    def _XGB(self) -> tuple[optuna.Study, dict]:
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
            "enable_categorical": True if self.cat_features is not None else False,
            "seed": self.random_state,
            "verbosity": 0,
        }
        return self._optimizer(xgb.XGBRegressor, param_space, static_params)
    ###########################################################################

    ###########################################################################
    """
    LightGBM regressor
    https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html
    """
    def _LGB(self) -> tuple[optuna.Study, dict]:
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
            "random_state": self.random_state,
            "verbose": -1,
            "n_jobs": -1,
        }
        return self._optimizer(LGBMRegressor, param_space, static_params)
    ###########################################################################

    ###########################################################################
    """
    CatBoost regression
    https://catboost.ai/en/docs/concepts/python-reference_catboostregressor
    """
    def _CAT(self) -> tuple[optuna.Study, dict]:
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
            "cat_features": self.cat_features,
            "random_seed": self.random_state,
            "verbose": 0,
        }
        return self._optimizer(CatBoostRegressor, param_space, static_params)
    ###########################################################################
