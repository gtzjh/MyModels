[English version](#English-Documentation)

[中文版本](#中文使用说明)

# English Documentation

**Store my machine learning and SHAP (SHapley Additive exPlanations) codes.**

*For regression task only currently.*

**DO REMEMBER: All models are wrong, but some are useful.**

## Models Supported

1. [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)

2. [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

3. [LightGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html)

4. [Catboost](https://catboost.ai/en/docs/concepts/python-reference_catboostregressor)

## 1. Prepare environment (On Windows platform)

*About 1.75 GiB storage would be used.*

conda (**Recommended**)

```bash
conda env create -f env.yml -n mymodels
```

```bash
conda activate mymodels
```

pip

```bash
pip install -r env.yml
```

# On-going

- [x] Test on Linux (64-bit)

- [x] Plot SHAP results

- [ ] Support categorical features (CatBoost is Recommended).

- [ ] GPU acceleration for training, testing, and SHAP.

- [ ] Adopt to the classification tasks.

******

# 中文使用说明

存储我常用的机器学习模型，并使用Optuna进行贝叶斯调参。用最少的时间完成机器学习任务。

*目前仅支持回归任务。*

**请记住：所有模型都是错的，但有一些是有用的。**

## 支持的模型

1. [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)

2. [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

3. [LightGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html)

4. [Catboost](https://catboost.ai/en/docs/concepts/python-reference_catboostregressor)

## 环境准备（Windows平台，其余平台同理）

*环境安装大约使用1.75 GiB存储空间*

conda (**推荐使用conda安装**)

```bash
conda env create -f env.yml -n mymodels
```

```bash
conda activate mymodels
```

pip

```bash
pip install -r env.yml
```

## 使用

### 模型训练及验证

运行主程序文件 `main.py`。

根据自己需要修改 `main.py` 第8 - 第13行。

```python
model = "cat"                # 模型选择: "lgb", "cat", "rf", "dt"，分别代表 LightGBM，CatBoost，Random Forest，Decision Tree。
results_dir = "results/CAT"  # 所有结果保存的文件夹，可自行修改。默认为在 results文件夹下创建与模型缩写同名的文件夹。
trials = 50                  # 使用Optuna调参，执行多少个Trials，次数越多，将会遍历更多的超参数可能性。默认为50，如无必要无需修改。
shap_ratio = 0.1             # 使用多少样本计算SHAP值。默认为10%。增加这一值将会需要更长的运行时间，但可以得到相对更加精确的结果，反之亦然。
cross_valid = 6              # 【不用修改】使用Optuna调参，每个Trial进行多少次交叉验证。
random_state = 6             # 【不用修改】全局随机种子, for model training, cross validation turning, and testing.
```

### SHAP解释绘图

目前仅支持全局解释与局部解释。

先安装 R (4.4.0以上版本) 与 RStudio，

打开 `MyModels.Rproj` 工程文件。

打开 plot.R 绘图程序。

1. 设定工作目录，即 `setwd()`处，根据自己本地路径修改。应指向 `MyModels` 文件夹的路径。

2. `results_dir <-` 注意一定要为自己保存结果的文件夹，对应上述主程序中的的 `results_dir`

检查确定所有参数正确，一般来说仅需要修改上述两个参数。

运行该脚本。

结果解释：在`results_dir`对应的目录下，可见`.png`后缀图片。

`scatter_plot.png` 模型训练与测试精度。

`optimization_plot.png` 调参过程中每个Trial得到的模型验证精度以及截止至当前Trial的最优精度。

`global_explanation_plot.png` 全局解释，按照mean(|SHAP|)从上至下排序，代表该变量的重要性。

`local_explanation_plot.png` 局部解释，可用于探究每个变量的影响阈值及拐点。
