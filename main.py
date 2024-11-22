import numpy as np
import pandas as pd
from data.dataLoader import dataLoader
from SHAP import SHAP
from models import ml


model = "rf"                 # Model selection: "lgb", "cat", "rf", "dt".
results_dir = "results/RF"   # Where the results will be store in.
trials = 50                  # How many trials to execute in optuna hyperparameters turning.
test_ratio = 0.2             # Ratio for test in the whole dataset.
shap_ratio = 0.1             # Use 10% of the whole dataset for SHAP calculation.
cross_valid = 3              # Cross validation in optuna hyperparameters turning.
random_state = 0             # Global random state control, for model training, cross validation turning, and testing.


def main():
    ###########################################################################
    # Data preparing
    x_train, x_test, y_train, y_test = dataLoader(
        file_path = "data/data.csv",
        y_index = 0, 
        x_index_list = range(1, 16),
        test_ratio = test_ratio
    )
    ###########################################################################

    ###########################################################################
    # Execute machine learning
    best_model = ml(
        x_train, x_test, y_train, y_test,  # Load data set.
        model = model,
        cv = cross_valid, 
        random_state = random_state,
        trials = trials,
        results_dir = results_dir,
        cat_features = None
    )
    ###########################################################################

    ###########################################################################
    # SHAP explanation
    # Sample the data set
    np.random.seed(random_state)
    all_data = pd.concat([x_train, x_test])
    shap_data = all_data.loc[
        np.random.choice(
            all_data.index, 
            int(len(all_data) * shap_ratio),
            replace = False
        )]

    _, _shap_values, _interaction = SHAP(
        best_model, shap_data, results_dir, explainer = "tree"
    )
    ###########################################################################
    
    return None


if __name__ == "__main__":
    main()