import numpy as np
import pandas as pd
from data.dataLoader import dataLoader
from myshap import myshap


from models import regressor

file_path = "data/data.csv"         # Where to load data
y_index = 0                         # Choose the index as dependency (y)
x_index_list = range(1, 16)         # Choose the index as independency (x)
model = "lgb"                       # Model selection: "lgb", "cat", "rf", "dt", "gbdt".
results_dir = "results/"            # Use the model name as the results dir, you can also pass the pathlib object
trials = 100                        # How many trials to execute in optuna hyperparameters turning.
test_ratio = 0.3                    # Ratio for test in the whole dataset.
shap_ratio = 0.3                    # Use 30% of the whole dataset for SHAP calculation.
cross_valid = 5                     # Cross validation in optuna hyperparameters turning.
random_state = 0                    # Global random state control, for model training, cross validation turning, and testing.


def main():
    ###########################################################################
    # Data preparing
    x_train, x_test, y_train, y_test = dataLoader(
        file_path = file_path,
        y_index = 0, 
        x_index_list = x_index_list,
        test_ratio = test_ratio,
        random_state = random_state
    )
    ###########################################################################

    ###########################################################################
    # Execute machine learning
    best_model = regressor(
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
    main()