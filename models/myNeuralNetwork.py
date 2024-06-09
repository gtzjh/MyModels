import numpy as np
import pandas as pd
import optuna
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score, KFold
from optuna.visualization import plot_optimization_history
from pathlib import Path


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

    # 以6折交叉验证的结果作为需要优化的返回值
    cv_obj = KFold(n_splits = cv, shuffle = True, random_state = random_state)
    def objective(trial):
        #######################################################################
        param = {
            "hidden_layer_sizes": (100, 100, 100),
            "max_iter": 1000,
            "early_stopping": True,
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-5, 1e-1, log=True),
        }
        #######################################################################

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

    # 贝叶斯调参
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

    # Output the best model, parameters, and accuracy.
    return (best_model, best_params, accuracy)


if __name__ == "__main__":
    NN()