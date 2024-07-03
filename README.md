# MyModels

[中文版本](中文使用说明.md)

[English version](README.md)

Store my machine learning and SHAP (SHapley Additive exPlanations) codes.

**DO REMEMBER: All models are wrong, but some are useful.**


# Models

1.  [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)

2.  [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

3.  [LightGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html)

4.  [Catboost](https://catboost.ai/en/docs/concepts/python-reference_catboostregressor)

*About 1.75 GiB storage would be used.*

## 1. Prepare environment (On Windows platform)

conda (**Recommended**)

``` bash
conda env create -f env.yml -n mymodels
```

``` bash
conda activate mymodels
```

pip

``` bash
pip install -r env.yml
```

# To-do

-   [x] Test on Linux (64-bit)

-   [x] Plot SHAP results

-   [ ] Support categorical features (CatBoost is Recommended).

-   [ ] GPU acceleration for training, testing, and SHAP.

-   [ ] Adopt to the classification tasks.

# Log

## 2024.07.03

1. Add figures of results.

2. Change the stop of installing environment.

## 2024.07.02

1. Due to the initial parameters are randomly set, some warnings may occur when implement CatBoost. Just ignore it or repeat.

2. The categorical features supporting must be rewriten.

## 2024.06.30

1. Output the accuracy, best parameters, optimization, and shap results.

## 2024.06.15

1.  Remove Autogluon.

2.  Modify NN module.

3.  Tested on linux (Ubuntu 22.04, WSL2)

## 2024.06.09

1.  Added lightgbm

## 2024.06.09

1.  Found that the KernelExplainer can not calculate interaction values.

2.  Finished the tree based model, and finished the test.

3.  Plan to extract some same codes from each seperated .py file, and rebuild the whole `models` module.

4.  The optuna should accomadate the consistent parameters and pass it to the final best parameters set, but now they were all ignored.

*Created by [Junhong](https://github.com/gtzjh). All rights reserved.*
