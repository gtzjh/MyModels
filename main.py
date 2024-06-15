import numpy as np
import pandas as pd
from data.dataLoader import dataLoader
from SHAP import SHAP
from models import ml, NN
from time import time


def main():
    ###########################################################################
    # Data preparing
    x_train, x_test, y_train, y_test = dataLoader(
        file_path = "data/data.csv",
        y_index = 0, 
        x_index_list = range(1, 16)
    )
    ###########################################################################

    ###########################################################################
    # Execute machine learning
    model, params, accuracy = ml(
        x_train, x_test, y_train, y_test,  # Load data set.
        model = "rf",                      # Model selection: "lgb", "cat", "rf", "dt".
        cv = 6,                            # Cross validation in optuna hyperparameters turning.
        random_state = 6,                  # Global random state control, for model training, cross validation turning, and testing.
        trials = 100,                      # How many trials to execute in optuna hyperparameters turning.
        results_dir = "results/RF"         # In which the optimazation results storing in.
    )
    print(params)
    print(accuracy)
    ###########################################################################

    ###########################################################################
    # SHAP explanation
    # Sample the data set
    np.random.seed(6)  # Use the random state for consistent results
    all_data = pd.concat([x_train, x_test])
    shap_data = all_data.loc[
        np.random.choice(
            all_data.index, 
            int(len(all_data) * 0.1),  # Use 10% of the whole dataset for SHAP calculation.
            replace = False
        )]

    _, shap_values, interaction = SHAP(
        model, shap_data, explainer = "tree"
    )

    print(shap_data)    # Data set used in SHAP
    print(shap_values)  # Local explanation
    print(interaction)  # Interaction
    ###########################################################################
    
    return None


if __name__ == "__main__":
    main()