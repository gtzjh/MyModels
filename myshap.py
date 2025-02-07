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
    plt.savefig(results_dir.joinpath('shap_summary.jpg'), dpi = 500)
    plt.close()


    # Partial Dependency Plot
    """
    注意: Partial Dependency Plot(PDP)的纵坐标是预测值,横坐标是自变量
    SHAP自带的scatter plot是基于SHAP值的,纵坐标是SHAP值,横坐标是自变量
    SHAP自带的dependence plot是基于SHAP值的,纵坐标是SHAP值,横坐标是自变量
    """
    results_dir.joinpath("PDP").mkdir(parents = True, exist_ok = True)  # Create the dir if not exists
    for _feature_name in shap_data.columns:
        shap.partial_dependence_plot(
            _feature_name,
            best_model.predict,
            shap_data,
            model_expected_value = True,
            feature_expected_value = True,
            ice = False,
            show = False
        )
        plt.tight_layout()
        plt.savefig(results_dir.joinpath("PDP").joinpath(_feature_name + '.jpg'), dpi = 500)
        plt.close()

    return None
