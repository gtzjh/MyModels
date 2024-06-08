# MyModels

Store my machine learning and SHAP (SHapley Additive exPlanations) codes.

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
conda create --name mymodels --file requirements.txt
conda activate mymodels
```

### Use pip

```bash
pip install -r requirements.txt
```

# Upcomming

1. GPU acceleration for training, testing, and explaining.

2. Adopt to the classification tasks.

# Attention

**Please do not install ray for faster training (In the autogluon recommandation)**
**Do not execute:**

```bash
mamba install -c conda-forge "ray-tune >=2.6.3,<2.7" "ray-default >=2.6.3,<2.7"  # Do not install it
```