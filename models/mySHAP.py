import numpy as np
import pandas as pd
import shap


"""
Parameters:
  1. model used for SHAP explaination
  2. data used for SHAP calculation
  3. The precentage of data used for SHAP calculation, 30% by default
不需要使用完整的测试数据集
"""


def exeSHAP(shap_model, shap_data, explainer = "tree", _prop = 0.3, _random_state = 6):
    assert isinstance(shap_data, pd.DataFrame)
    assert explainer == "tree" or explainer == "kernel"

    explainer = shap.TreeExplainer(shap_model)

    #  Use the random state for consistent results
    np.random.seed(_random_state)
    shap_data_selected = shap_data.loc[
        np.random.choice(
            shap_data.index, 
            int(len(shap_data) * _prop),  # 随机抽样一部分的样本
            replace = False
        )]
    
    shap_interaction_values = np.array(
        explainer.shap_interaction_values(shap_data_selected)
    )


    # 交互作用
    interaction_dict = dict()
    for i in range(0, len(shap_data_selected.columns)):
        for k in range(0, i + 1):
            # A column is represented an interaction pair
            _column_name = shap_data_selected.columns[i] + "_" + shap_data.columns[k]
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

    
    # 单因子分析 实际上就是交互作用的 2 + 3维 的对角线
    shap_values_dict = dict()
    for i in range(0, len(shap_data_selected.columns)):
        _shap_values_column_name = shap_data_selected.columns[i]
        _shap_values_column = shap_interaction_values[:, i, i]
        shap_values_dict.update({
            _shap_values_column_name: _shap_values_column
        })
    shap_values_df = pd.DataFrame(shap_values_dict)


    return {
        "shap_data": shap_data_selected,    # SHAP数据
        "shap_value": shap_values_df,       # 每个样本对应每个特征的值
        "shap_interaction": interaction_df  # 交互作用
    }
