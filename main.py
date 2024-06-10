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
        x_train, x_test, y_train, y_test,
        model = "rf",
        cv = 6, 
        random_state = 42, 
        trials = 100, 
        results_dir = "results/RF"
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
    start = time.now()
    main()
    end = time.now()
    print("Elapse: ", (round(end-start)/3600, 2))