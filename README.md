# MyModels

Store my machine learning and SHAP (SHapley Additive exPlanations) codes.

**DO REMEMBER:**

[George E. P. Box: ](https://en.wikipedia.org/wiki/George_E._P._Box) **All models are wrong, but some are useful.** 

# Models

1. [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)

2. [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

3. [LightGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html)

4. [Catboost](https://catboost.ai/en/docs/concepts/python-reference_catboostregressor)

5. [Neural Network (MLP)](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)

   (Must use the `KernelExplainer`)

# Usage

## 1. Prepare python environments

- Platform: 
  - Windows 10 64-bit or later. (Tested on Windows 11 professional 64-bit)
  - Linux (Tested on Ubuntu 22.04)

- Python version: 3.10.x

*About 9 GiB storage would be used.*

### 1.1 Use conda (Recommended)

```bash
conda create -n mymodels -f requirements.txt -c conda-forge -y  # Install packages
conda activate mymodels                                         # Activate environment
```

### 1.2 Use pip

```bash
pip install -r requirements.txt
```

## 2. Execute

In bash:

```bash
cd MyModels     # Entry the work dir
python main.py  # Execute
```


# To-do

- [x] Support categorical features (Recommend: CatBoost).

- [ ] GPU acceleration for training, testing, and SHAP.

- [ ] Adopt to the classification tasks.

- [ ] Test on Linux (64-bit), MacOS (Apple M1 silicon or later).

# Attention

**Please do not install ray for faster training (In the autogluon recommandation)**

**Do not execute:**

```bash
mamba install -c conda-forge "ray-tune >=2.6.3,<2.7" "ray-default >=2.6.3,<2.7"
```

# Logs

## 2024.06.09

1. Added lightgbm

## 2024.06.09

1. Found that the KernelExplainer can not calculate interaction values

2. Finished the tree based model, and finished the test

3. Plan to extract some same codes from each seperated .py file, and rebuild the whole `models` module.

4. The optuna should accomadate the consistent parameters and pass it to the final best parameters set, but now they were all ignored.


Powered [gtzjh](https://github.com/gtzjh). All rights reserved.