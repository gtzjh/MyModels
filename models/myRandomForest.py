import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score, KFold
from optuna.visualization import plot_optimization_history


def RF(
        x_train, x_test, y_train, y_test,   # Input train and test data
        cv = 6,                             # Cross-validation for 6 times
        random_state = 42,                  # Random state is 42
        trials = 100,                       # Execute 100 times in optuna
    ):

    # 以6折交叉验证的结果作为需要优化的返回值
    cv_obj = KFold(n_splits = cv, shuffle = True, random_state = random_state)
    def objective(trial):
        
        #######################################################################
        # Parameter grid
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 3000, step = 100),
            "max_depth": trial.suggest_int("max_depth", 1, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
            "random_state": random_state,
        }
        #######################################################################
        
        preditor = RandomForestRegressor(**param)
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
    opt_history_df = pd.DataFrame(data = {
        "trials": np.array(fig_obj.data[0]["x"]),
        "obj": np.array(fig_obj.data[0]["y"]),
        "best": np.array(fig_obj.data[1]["y"])
    })

    opt_history_df.set_index("trials", inplace = True)
    opt_history = pd.concat([opt_history, opt_history_df], axis = 1)
    opt_history.to_csv("results/RF/optimization.csv", encoding = "utf-8")

    # Return the best trial and parameters
    trial = study.best_trial
    best_params = trial.params
    best_params.update({
        "random_state": random_state,
    })

    # Train a model base upon the optimal parameters on the whole train data.
    best_model = RandomForestRegressor(**best_params)
    best_model.fit(x_train, y_train)
    y_pred = best_model.predict(x_test)
    best_params.update({
        "R2": r2_score(y_test, y_pred),
        "RMSE": root_mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "MAPE": mean_absolute_percentage_error(y_test, y_pred)
    })
    
    print("--------------------------------------------------------------------")
    print(best_params)
    print("--------------------------------------------------------------------")

    # Output the best model, parameters, and accuracy.
    return (best_model, best_params)


if __name__ == "__main__":
    RF()


