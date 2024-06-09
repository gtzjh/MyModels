# import numpy as np
# import pandas as pd
from data.dataLoader import dataLoader
from models.myRandomForest import RF
from models.myCatBoost import CAT
from models.myDecisionTree import DT
# from models.myNeuralNetwork import NN
from models.mySHAP import SHAP



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
    model, params, accuracy = CAT(
        x_train, x_test, y_train, y_test,
        cv = 6, random_state = 42, trials = 10, results_dir = "results/CAT"
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
    main()