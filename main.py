import numpy as np
import pandas as pd
from data.dataLoader import dataLoader
import shap
import matplotlib.pyplot as plt
from models import ml
from pathlib import Path


shap.initjs()
plt.rc('font', family = 'Times New Roman')


model = "rf"                        # Model selection: "lgb", "cat", "rf", "dt", "gbdt".
results_dir = Path("results/").joinpath(model) # Use the model name as the results dir
trials = 50                         # How many trials to execute in optuna hyperparameters turning.
test_ratio = 0.3                    # Ratio for test in the whole dataset.
shap_ratio = 0.3                    # Use 30% of the whole dataset for SHAP calculation.
cross_valid = 5                     # Cross validation in optuna hyperparameters turning.
random_state = 0                    # Global random state control, for model training, cross validation turning, and testing.


def main():
    ###########################################################################
    # Data preparing
    x_train, x_test, y_train, y_test = dataLoader(
        file_path = "data/data.csv",
        y_index = 0, 
        x_index_list = range(1, 15),
        test_ratio = test_ratio
    )
    ###########################################################################

    ###########################################################################
    # Execute machine learning
    best_model = ml(
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
    
    # Summary plot
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer(shap_data)
    shap.summary_plot(shap_values, shap_data, show = False)
    plt.tight_layout()
    plt.savefig(results_dir.joinpath('shap_summary.jpg') , dpi = 500)
    plt.savefig(results_dir.joinpath('shap_summary.pdf') , dpi = 500)
    plt.close()


    # Dependency plot
    # Create the dir if not exists
    results_dir.joinpath("PDP").mkdir(parents = True,
                                        exist_ok = True)
    for _feature_name in shap_data.columns:
        shap.plots.scatter(shap_values[:, _feature_name],
                            color = "steelblue", alpha = 0.4, dot_size = 27,
                            show = False)
        plt.tight_layout()
        plt.savefig(results_dir.joinpath("PDP").joinpath(_feature_name + '.jpg') , dpi = 500)
        plt.savefig(results_dir.joinpath("PDP").joinpath(_feature_name + '.pdf') , dpi = 500)
        plt.close()

    return None


if __name__ == "__main__":
    main()