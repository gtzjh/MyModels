# MyModels

[中文版本](使用说明.md)

[English version](README.md)

Store my machine learning and SHAP (SHapley Additive exPlanations) codes.

**DO REMEMBER: All models are wrong, but some are useful.**

## Models Supported

1.  [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)

2.  [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

3.  [LightGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html)

4.  [Catboost](https://catboost.ai/en/docs/concepts/python-reference_catboostregressor)


## 1. Prepare environment (On Windows platform)

*About 1.75 GiB storage would be used.*

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

# On-going

-   [x] Test on Linux (64-bit)

-   [x] Plot SHAP results

-   [ ] Support categorical features (CatBoost is Recommended).

-   [ ] GPU acceleration for training, testing, and SHAP.

-   [ ] Adopt to the classification tasks.

