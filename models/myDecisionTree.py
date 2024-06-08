import numpy as np
import pandas as pd
import optuna
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score
from optuna.visualization import plot_optimization_history
from data.dataLoader import dataLoader


# import warnings
# warnings.filterwarnings("ignore")


def DT():
    opt_history = pd.DataFrame()
    model_results = []

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
        def objective(trial):
            param = {
                "max_depth": trial.suggest_int("max_depth", 2, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_features": trial.suggest_float("max_features", 0.5, 1),
                "random_state": 42,
            }
            
            preditor = DecisionTreeRegressor(**param)
            cv_r2 = cross_val_score(
                preditor, 
                x_train, y_train, 
                scoring = "r2",
                cv = 6,
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
        opt_history_df = pd.DataFrame(data = {
            "trials": np.array(fig_obj.data[0]["x"]),
            grid_name[7:] + "m" + "_obj": np.array(fig_obj.data[0]["y"]),
            grid_name[7:] + "m" + "_best": np.array(fig_obj.data[1]["y"])
        })
        opt_history_df.set_index("trials", inplace = True)
        opt_history = pd.concat([opt_history, opt_history_df], axis = 1)
        opt_history.to_csv("Results\\DT\\opt_history.csv", encoding = "utf-8")

        # 返回最优的trial和参数组合
        trial = study.best_trial
        best_params = trial.params
        best_params.update({
            "random_state": 42,
        })

        # 将最优参数在训练集+验证集的基础上重新训练一个模型用于预测
        best_model = DecisionTreeRegressor(**best_params)
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

        # 每跑完一个尺度都要保存
        model_results.append(best_params)
        model_results_df = pd.DataFrame.from_records(model_results)
        model_results_df.set_index("Scale", inplace=True)
        model_results_df.to_csv("Results\\DT\\DecisionTree_results.csv", encoding="utf-8")

        # 节约内存（其实作用不大）
        del study
        del fig_obj
        del opt_history_df
        del trial
        del best_model
        del best_params
        del y_pred
        del model_results_df
        
    return None



if __name__ == "__main__":
    DT()


