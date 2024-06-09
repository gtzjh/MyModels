import numpy as np
import pandas as pd
import optuna
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score, KFold
from optuna.visualization import plot_optimization_history
from pathlib import Path


def DT(
        x_train, x_test, y_train, y_test,   # Input train and test data
        cv = 6,                             # Cross-validation for 6 times
        random_state = 42,                  # Random state is 42
        trials = 100,                       # Execute 100 times in optuna
        results_dir = "results/DT"          # The dir to store the optimization results
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
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_float("max_features", 0.5, 1),
        }
        
        #######################################################################
        
        preditor = DecisionTreeRegressor(**param)
        cv_r2 = cross_val_score(
            preditor, 
            x_train, y_train, 
            scoring = "r2",
            cv = 6,
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

    # Return the best trial and parameters
    trial = study.best_trial
    best_params = trial.params

    # Train a model base upon the optimal parameters on the whole train data.
    best_model = DecisionTreeRegressor(**best_params)
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
    DT()


