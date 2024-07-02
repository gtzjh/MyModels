# MyModels

Store my machine learning and SHAP (SHapley Additive exPlanations) codes.

**DO REMEMBER:**

[**All models are wrong, but some are useful.**](https://en.wikipedia.org/wiki/George_E._P._Box)

# Models

1.  [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)

2.  [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

3.  [LightGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html)

4.  [Catboost](https://catboost.ai/en/docs/concepts/python-reference_catboostregressor)

    (Must use the `KernelExplainer`)

# Usage

-   Platform:
    -   Windows 10 64-bit or later. (Tested on Windows 11)
    -   Linux (Tested on Ubuntu 22.04 WSL2)
-   Python version: 3.10.x

*About 1.75 GiB storage would be used.*

## 1. Prepare environment

conda (**Recommended**)

*On Windows*

``` bash
conda create --name mymodels --file windows.txt -c conda-forge -y
```

*Replace `windows.txt` to `linux.txt` on linux platform.*

``` bash
conda activate mymodels
```

pip

``` bash
pip install -r windows.txt  # The same replacement to above.
```

## 2. Execute

# To-do

-   [ ] Support categorical features (CatBoost is Recommended).

-   [x] Test on Linux (64-bit)

-   [x] Plot SHAP results

-   [ ] GPU acceleration for training, testing, and SHAP.

-   [ ] Adopt to the classification tasks.

# Logs


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
