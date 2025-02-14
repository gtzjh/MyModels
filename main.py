import numpy as np
import pandas as pd
from dataLoader import dataLoader
from myshap import myshap
from myregressors import regr


def main(file_path, y, x_list, cat_features, model,
         results_dir, trials, test_ratio, shap_ratio, cross_valid, random_state):
    ###########################################################################
    # Data preparing
    x_train, x_test, y_train, y_test = dataLoader(
        file_path = file_path,
        y = y,
        x_list = x_list,
        cat_features = cat_features,
        test_ratio = test_ratio,
        random_state = random_state
    )
    ###########################################################################

    ###########################################################################
    # Execute machine learning
    optimal_model = regr(
        x_train, x_test, y_train, y_test,
        model = model,
        cv = cross_valid,
        random_state = random_state,
        trials = trials,
        results_dir = results_dir,
        # cat_features = cat_features
    )
    ###########################################################################

    ###########################################################################
    # SHAP test
    np.random.seed(random_state)
    all_data = pd.concat([x_train, x_test])
    shap_data = all_data.loc[
        np.random.choice(
            all_data.index,
            int(len(all_data) * shap_ratio),
            replace = False
        )]
    myshap(optimal_model, model, shap_data, results_dir)
    ###########################################################################

    return None


if __name__ == "__main__":
    for i in [
        # "dt", "rf", "gbdt", 
        #"ada", 
        "xgb", "lgb", "cat"
    ]:
        """
        file_path: Where to load data
        y: Choose the index as dependency (y), you can also pass string of variables' name
        x_list: Choose the index as independency (x), you can also pass a list of string of variables' name
        cat_features: Choose the index as categorical features (x), you can also pass a list of string of variables' name
        model:
            - "dt": Decision Tree
            - "rf": Random Forest
            - "gbdt": Gradient Boosting Decision Tree
            - "ada": AdaBoost  AdaBoost can be slow in SHAP values computation 'cause the KernelExplainer will be implemented.
            - "xgb": XGBoost
            - "lgb": LightGBM
            - "cat": CatBoost
        results_dir: Use the model name as the results dir, you can also pass the pathlib object
        trials: How many trials to execute in optuna hyperparameters turning.
        test_ratio: Ratio for test in the whole dataset.
        shap_ratio: Use 10% of the whole dataset for SHAP calculation.
        cross_valid: Cross validation in optuna hyperparameters turning.
        random_state: Global random state control, for model training, cross validation turning, and testing.
        """
        main(
            file_path = "data2.csv",
            y = "y",
            x_list = list(range(1, 16)),
            cat_features = None,
            model = i,
            results_dir = "results/" + i,
            trials = 10,
            test_ratio = 0.3,
            shap_ratio = 0.3,
            cross_valid = 5,
            random_state = 0,
        )
        print(f"Finished {i}")