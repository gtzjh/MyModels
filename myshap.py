import pandas as pd
import shap
import matplotlib.pyplot as plt
import pathlib

shap.initjs()
plt.rc('font', family = 'Times New Roman')


def myshap(model, model_type, shap_data, results_dir):
    """
    SHAP Visualization Functions Comparison:
    
    1. shap.partial_dependence_plot:
    - Shows average marginal effect of a feature on model output
    - Similar to traditional Partial Dependence Plot (PDP)
    - X-axis: Feature values, Y-axis: Predicted values
    
    2. shap.dependence_plot:
    - Shows relationship between feature values and SHAP values
    - Automatically detects interactions (color-codes 2nd influential feature)
    - X-axis: Feature values, Y-axis: SHAP values
    
    3. shap.scatter_plot:
    - Basic SHAP value visualization
    - Requires manual specification of x/y axes
    - No automatic interaction detection
    
    Key Differences:
    | Function                | Data Source     | SHAP Values | Auto-Interaction | Output Scale |
    |-------------------------|-----------------|-------------|------------------|--------------|
    | partial_dependence_plot | Raw feature     | ❌          | ❌              | Prediction   |
    | dependence_plot         | Raw + SHAP      | ✅          | ✅              | SHAP         |
    | scatter_plot            | SHAP values     | ✅          | ❌              | SHAP         |
    
    Recommendation: Use dependence_plot for most cases, scatter_plot for custom combinations
    """
    # Check input
    assert isinstance(shap_data, pd.DataFrame)
    assert isinstance(results_dir, pathlib.Path) or isinstance(results_dir, str)
    results_dir = pathlib.Path(results_dir)
    results_dir.mkdir(parents = True, exist_ok = True)

    # Set different explainers for different models
    if model_type in ["svr", "knr", "mlp", "ada"]:
        # For KernelExplainer, need to pass prediction function and data
        explainer = shap.KernelExplainer(model.predict, shap_data)
        shap_values = explainer.shap_values(shap_data)
    else:
        # For TreeExplainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(shap_data)

    # Summary plot
    shap.summary_plot(shap_values, shap_data, show = False)
    plt.tight_layout()
    plt.savefig(results_dir.joinpath('shap_summary.jpg'), dpi = 500)
    plt.close()

    # Partial Dependency Plot
    results_dir.joinpath("partial_dependence_plots") \
               .mkdir(parents = True, exist_ok = True)
    for _feature_name in shap_data.columns:
        shap.partial_dependence_plot(
            _feature_name,
            model.predict,
            shap_data,
            model_expected_value = True,
            feature_expected_value = True,
            ice = False,
            show = False
        )
        plt.tight_layout()
        plt.savefig(results_dir.joinpath("partial_dependence_plots")\
                    .joinpath(_feature_name + '.jpg'), dpi = 500)
        plt.close()
    
    # Dependence Plot
    results_dir.joinpath("dependence_plots") \
               .mkdir(parents = True, exist_ok = True)
    for _feature_name in shap_data.columns:
        shap.dependence_plot(
            _feature_name,
            shap_values,
            shap_data,
            show = False
        )
        plt.tight_layout()
        plt.savefig(results_dir.joinpath("dependence_plots")\
                    .joinpath(_feature_name + '.jpg'), dpi = 500)
        plt.close()

    # There is no scatter plot here.
    return None
