from data.dataLoader import dataLoader
from SHAP import SHAP
from models import ml
from time import time



"""
Do not execute NN explainer now, some error occur.
"""
def main():
    # Data preparing
    x_train, x_test, y_train, y_test = dataLoader(
        file_path = "data/data.csv",
        y_index = 0, 
        x_index_list = range(1, 16)
    )

    # Execute machine learning
    model, params, accuracy = ml(
        x_train, x_test, y_train, y_test,  # Load data set.
        model = "lgb",                     # Model selection: "lgb", "cat", "rf", "dt".
        cv = 6,                            # Cross validation in optuna hyperparameters turning.
        random_state = 6,                  # Global random state control, for model training, cross validation turning, and testing.
        trials = 100,                       # How many trials to execute in optuna hyperparameters turning.
        results_dir = "results/LGB"        # In which the optimazation results storing in.
    )
    print(params)
    print(accuracy)


    # SHAP explanation
    shap_data, shap_values, interaction = SHAP(
        model, x_test.iloc[0:200], explainer = "tree"
    )

    print(shap_data)    # Data set used in SHAP
    print(shap_values)  # Local explanation
    print(interaction)  # Interaction
    
    return None


if __name__ == "__main__":
    start = time()
    main()
    end = time()
    print("Elapse: %f hours.".format(round((end-start)/3600, 2)))