import numpy as np
import pandas as pd
import shap


"""
Parameters:
  1. model used for SHAP explaination
  2. data used for SHAP calculation
"""
def mySHAP(shap_model, shap_data, explainer = "tree"):
    assert isinstance(shap_data, pd.DataFrame)
    assert explainer == "tree" or explainer == "kernel"
    if explainer == "tree":
        explainer = shap.TreeExplainer(shap_model)
    elif explainer == "kernel":
        explainer = shap.KernelExplainer(shap_model)


    shap_interaction_values = np.array(
        explainer.shap_interaction_values(shap_data)
    )


    # 交互作用
    interaction_dict = dict()
    for i in range(0, len(shap_data.columns)):
        for k in range(0, i + 1):
            # A column is represented an interaction pair
            _column_name = shap_data.columns[i] + "_" + shap_data.columns[k]
            _column = shap_interaction_values[:, i, k]
            interaction_dict.update({
                _column_name: _column
            })
    
    interaction_df = pd.DataFrame(interaction_dict)

    # Calculate the mean(|SHAP|) as the order of the dataframe
    mean_interaction = interaction_df.abs().mean(axis = 0)
    mean_interaction.sort_values(
        ascending = False,
        inplace = True
    )
    order_list = mean_interaction.index.to_list()
    interaction_df = interaction_df[order_list]

    
    # 单因子分析 实际上就是交互作用的 2 + 3 维 的对角线
    shap_values_dict = dict()
    for i in range(0, len(shap_data.columns)):
        _shap_values_column_name = shap_data.columns[i]
        _shap_values_column = shap_interaction_values[:, i, i]
        shap_values_dict.update({
            _shap_values_column_name: _shap_values_column
        })
    shap_values_df = pd.DataFrame(shap_values_dict)


    return (
        shap_data,       # SHAP数据
        shap_values_df,  # 每个样本对应每个特征的值
        interaction_df   # 交互作用
    )
    
