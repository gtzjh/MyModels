import numpy as np
import pandas as pd
from data.dataLoader import dataLoader
from models.myRandomForest import RF
from models.mySHAP import mySHAP


def main():
    x_train, x_test, y_train, y_test = dataLoader(
        file_path = "data/data.csv",
        y_index = 0, 
        x_index_list = [3, 4, 5]
    )

    # Use random forest as an example
    model, params = RF(
        x_train, x_test, y_train, y_test,
        cv = 6, random_state = 42, trials = 100
    )

    # SHAP explanation
    shap_data, shap_values, interaction = mySHAP(
        model, x_test, explainer = "tree"
    )

    return None


if __name__ == "__main__":
    main()