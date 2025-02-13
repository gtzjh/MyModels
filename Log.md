# Log

## 20250213

Adjusted the README.md.

Added XGBoost and AdaBoost.

Add the dependency plot for SHAP.

## 20250207

Corrected the partial dependency plot (PDP).

Simplified the code.

Abandon the gpu support, for 3 reasons:

- Data moving from memory to gpu's memory will take more time.

- The dataset is not big enough for gpu to perform better than cpu.

- Some contradition will occur when using gpu, and not every type of model has perfect support on gpu.

## 20241206

Remove the R plotting code for scatter plot.

## 20241129

Remove the R plotting code for SHAP summary plot and partial dependency plot (PDP).

## 2024.11.27
1. Add GBDT.

2. Used another dataset for demonstration.

## 2024.10.10

1. Add Chinese version.

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