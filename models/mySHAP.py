import numpy as np
import pandas as pd
import shap


"""
Parameters:
  1. model used for SHAP explaination
  2. data used for SHAP calculation
"""
def SHAP(shap_model, shap_data, explainer = "tree"):
    assert isinstance(shap_data, pd.DataFrame)
    assert explainer == "tree" or explainer == "kernel"
    if explainer == "tree":
        explainer = shap.TreeExplainer(shap_model)
    elif explainer == "kernel":
        explainer = shap.KernelExplainer(shap_model)
    else:
        return KeyError


    shap_interaction_values = np.array(
        explainer.shap_interaction_values(shap_data)
    )
    data_columns = shap_data.columns  # The columns list of the shap data.

    ###########################################################################
    # Interaction values
    interaction_dict = dict()
    for i in range(0, len(data_columns)):
        for k in range(0, i + 1):
            # A column is represented an interaction pair.
            # The columns will be presented like: x1_x2,
            # and the repeat like x2_x1 will be dropped.
            _column_name = data_columns[i] + "_" + data_columns[k]
            _column_values = shap_interaction_values[:, i, k]
            interaction_dict.update({
                _column_name: _column_values
            })
    interaction = pd.DataFrame(interaction_dict)
    #######################################################

    #######################################################
    # Calculate the mean(|SHAP|) for all the single and interaction factors.
    # The results contain both single and interaction.
    mean_interaction = interaction.abs().mean(axis = 0)
    mean_interaction.sort_values(ascending = False, inplace = True)
    order_list = mean_interaction.index.to_list()
    interaction = interaction[order_list]
    ###########################################################################

    ###########################################################################
    # Local explanation, actually the i, i values in the interaction matrix.
    # This is different to the shap value calculation in common, reference from:
    # https://zhuanlan.zhihu.com/p/103370775
    shap_values_dict = dict()
    for i in range(0, len(data_columns)):
        shap_values_dict.update({
            # Column name  : Corresponding list of local shap value
            data_columns[i]: shap_interaction_values[:, i, i]
        })
    shap_values = pd.DataFrame(shap_values_dict)
    ###########################################################################


    return (
        shap_data,    # Data set used for calculating SHAP.
        shap_values,  # Local explanation.
        interaction   # Interaction explanation.
    )
    
