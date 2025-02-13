import numpy as np
import pandas as pd
from dataLoader import dataLoader
from myshap import myshap
from myregressors import regr


def main(file_path, y, x_list, model, results_dir, trials, test_ratio, shap_ratio, cross_valid, random_state):
    # Data preparing
    x_train, x_test, y_train, y_test = dataLoader(
        file_path = file_path,
        y = y,
        x_list = x_list,
        test_ratio = test_ratio,
        random_state = random_state
    )
    ###########################################################################

    ###########################################################################
    # Execute machine learning
    best_model = regr(
        x_train, x_test, y_train, y_test,
        model = model,
        cv = cross_valid,
        random_state = random_state,
        trials = trials,
        results_dir = results_dir,
        cat_features = None
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
    myshap(best_model, shap_data, results_dir)
    ###########################################################################

    return None


if __name__ == "__main__":
    main(
        file_path = "data.csv",              # Where to load data
        y = "y",                             # Choose the index as dependency (y)，, you can also pass string of variables' name
        x_list = list(range(1, 15)),         # Choose the index as independency (x), you can also pass a list of string of variables' name
        model = "cat",                       # Model selection: "lgb", "cat", "rf", "dt", "gbdt".
        results_dir = "results/",            # Use the model name as the results dir, you can also pass the pathlib object
        trials = 200,                        # How many trials to execute in optuna hyperparameters turning.
        test_ratio = 0.3,                    # Ratio for test in the whole dataset.
        shap_ratio = 0.3,                    # Use 30% of the whole dataset for SHAP calculation.
        cross_valid = 5,                     # Cross validation in optuna hyperparameters turning.
        random_state = 0,                    # Global random state control, for model training, cross validation turning, and testing.
    )