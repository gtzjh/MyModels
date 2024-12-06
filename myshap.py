import pandas as pd
import shap
import matplotlib.pyplot as plt
import pathlib

shap.initjs()
plt.rc('font', family = 'Times New Roman')


def myshap(best_model, shap_data, results_dir):
    assert isinstance(shap_data, pd.DataFrame)
    assert isinstance(results_dir, pathlib.Path) or isinstance(results_dir, str)
    results_dir = pathlib.Path(results_dir)
    results_dir.mkdir(parents = True, exist_ok = True)

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
    results_dir.joinpath("PDP").mkdir(parents = True, exist_ok = True)
    for _feature_name in shap_data.columns:
        shap.plots.scatter(shap_values[:, _feature_name],
                           color = "#4682B4", alpha = 0.4, dot_size = 40,
                           show = False)
        plt.tight_layout()
        plt.savefig(results_dir.joinpath("PDP").joinpath(_feature_name + '.jpg') , dpi = 500)
        plt.savefig(results_dir.joinpath("PDP").joinpath(_feature_name + '.pdf') , dpi = 500)
        plt.close()

    return None

