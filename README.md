# MyModels

Store my machine learning and SHAP (SHapley Additive exPlanations) codes.

# Developing logs

## 2024.06.09

1. Found that the KernelExplainer can not calculate interaction values

2. Finished the tree based model, and finished the test

3. Plan to extract some same codes from each seperated .py file, and rebuild the whole `models` module.

4. The optuna should accomadate the consistent parameters and pass it to the final best parameters set, but now they were all ignored.

# Models

1. Neural Network (Must use the `KernelExplainer`)

2. Decision Tree

3. Random Forest

4. LightGBM

5. Catboost

# Usage

## Prepare python environments

### Use conda (Recommended)

```bash
conda create -n mymodels --file requirements.txt
conda activate mymodels
```

### Use pip

```bash
pip install -r requirements.txt
```

## Execute

-> main.py

# Upcomming

1. GPU acceleration for training, testing, and explaining.

2. Adopt to the classification tasks.

# Attention

**Please do not install ray for faster training (In the autogluon recommandation)**
**Do not execute:**

```bash
mamba install -c conda-forge "ray-tune >=2.6.3,<2.7" "ray-default >=2.6.3,<2.7"  # Do not install it
```