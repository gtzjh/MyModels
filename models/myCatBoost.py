import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score, KFold
from optuna.visualization import plot_optimization_history
from data.dataLoader import dataLoader


import warnings
warnings.filterwarnings("ignore")


def CAT():
    opt_history = pd.DataFrame()
    cat_results = []

    for grid_name in [
        "fishnet200",
        "fishnet300",
        "fishnet400",
        "fishnet500",
        "fishnet600",
        "fishnet700", 
        "fishnet800", 
        "fishnet900", 
        "fishnet1000"
    ]:
        x_train, x_test, y_train, y_test = dataLoader(grid_name)

        # 以6折交叉验证的结果作为需要优化的返回值
        cv_obj = KFold(n_splits = 6, shuffle = True, random_state = 6)
        def objective(trial):
            param = {
                "silent": True,
                "loss_function": "RMSE", # Catboost中不能使用 R2 作为 loss function
                "early_stopping_rounds": 20,
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
            
            preditor = CatBoostRegressor(**param)
            cv_r2 = cross_val_score(
                preditor, 
                x_train, y_train, 
                scoring = "r2",
                cv = cv_obj,
                n_jobs = -1,
                verbose = 0,
            )
            # print(np.mean(cv_r2))
            return np.mean(cv_r2)

        
        # 贝叶斯调参
        study = optuna.create_study(direction = "maximize")
        study.optimize(
            objective,
            n_trials = 100
        )
        
        # 将每次 trial 的验证精度（objective value and best value）保存，用于绘图
        fig_obj = plot_optimization_history(study)
        opt_history_df = pd.DataFrame(data={
            "trials": np.array(fig_obj.data[0]["x"]),
            grid_name[7:] + "m" + "_obj": np.array(fig_obj.data[0]["y"]),
            grid_name[7:] + "m" + "_best": np.array(fig_obj.data[1]["y"])
        })
        opt_history_df.set_index("trials", inplace=True)
        opt_history = pd.concat([opt_history, opt_history_df], axis=1)
        opt_history.to_csv("Results\\CAT\\opt_history.csv", encoding="utf-8")

        # 返回最优的trial和参数组合
        trial = study.best_trial
        best_params = trial.params
        best_params.update({
            "silent": True,
            "loss_function": "RMSE",
            "early_stopping_rounds": 20,
            "iterations": 2000,
        })

        # 将最优参数在训练集+验证集的基础上重新训练一个模型用于预测
        best_model = CatBoostRegressor(**best_params)
        best_model.fit(x_train, y_train)
        y_pred = best_model.predict(x_test)
        best_params.update({
            "Scale": grid_name[7:] + "m",
            "R2": r2_score(y_test, y_pred),
            "RMSE": root_mean_squared_error(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "MAPE": mean_absolute_percentage_error(y_test, y_pred)
        })
        print(best_params)

        #######################################################################
        # 每跑完一个尺度都要保存
        cat_results.append(best_params)
        cat_results_df = pd.DataFrame.from_records(cat_results)
        cat_results_df.set_index("Scale", inplace = True)
        cat_results_df.to_csv("Results\\CAT\\Accuracy.csv", encoding="utf-8")
        #######################################################################
    return None



if __name__ == "__main__":
    CAT()


