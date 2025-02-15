[English version](#English-Documentation)

[中文版本](#中文使用说明)


# On-going

# English Documentation

**Machine learning pipeline with automated hyperparameter tuning using Optuna and model interpretation with SHAP (SHapley Additive exPlanations).**

**Remember: All models are wrong, but some are useful. - George Box**

*Currently supports regression tasks only.*

Supported Models with Hyperparameter Optimization:

1. [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
2. [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
3. [Gradient Boosted Decision Trees (GBDT)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
4. [LightGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html)
5. [CatBoost](https://catboost.ai/en/docs/concepts/python-reference_catboostregressor)
6. [XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_api.html)
7. [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)
8. [Support Vector Regression (SVR)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
9. [K-Nearest Neighbors Regression (KNR)](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
10. [Multi-Layer Perceptron (MLP)](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)

## 0. Prerequisites

1. Python programming proficiency. Recommended resource: [Liao Xuefeng's Python Tutorial](https://liaoxuefeng.com/books/python/introduction/index.html) (Focus on sections 7-11 and 17)
2. Machine learning fundamentals. Recommended course: [CS229 by Andrew Ng](https://www.bilibili.com/video/BV1JE411w7Ub)
3. Environment management:
   - Conda/Pip package management
   - Terminal/Command Line proficiency
   - Git version control ([Example project](https://github.com/gtzjh/learngit))
   - Recommended IDE: Visual Studio Code

## 1. Environment Setup (Windows)

**Python 3.10 required**  
*Approximately 1.75 GiB disk space required*

Using Conda:
```bash
conda env create -f env-win.yml -n mymodels
conda activate mymodels
```

## 2. Usage

Configure parameters in `main.py`:

```python
file_path = "data.csv"              # Dataset path (CSV format)
y = "y"                             # Target variable name
x_list = list(range(1, 16))         # Feature indices/names
model = "cat"                       # Model selection: "dt", "rf", "gbdt", "xgb", "lgb", "cat", "ada", "svr", "knr", "mlp"
results_dir = "results/cat"         # Output directory (pathlib.Path compatible)
cat_features = ["x16", "x17"]       # List of categorical feature names (optional)
trials = 50                         # Optuna optimization trials
test_ratio = 0.3                    # Proportion of dataset to use as test set
shap_ratio = 0.3                    # Dataset proportion for SHAP analysis
cross_valid = 5                     # Cross-validation folds during optimization
random_state = 0                    # Random seed for reproducibility
```

Execute the pipeline:
```bash
python main.py
```



# 中文使用说明

存储我常用的机器学习模型，并使用Optuna进行贝叶斯调参。用最少的时间完成机器学习任务。

**请记住：所有模型都是错的，但有一些是有用的。**

*目前仅支持回归任务。*

本项目包含的模型：

1. [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
2. [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
3. [Gradient Boosted Decision Trees (GBDT)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
4. [LightGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html)
5. [Catboost](https://catboost.ai/en/docs/concepts/python-reference_catboostregressor)
6. [XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_api.html)
7. [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)
8. [Support Vector Regression (SVR)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
9. [K-Nearest Neighbors Regression (KNR)](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
10. [Multi-Layer Perceptron (MLP)](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)

## 0. 前置知识

1. 熟悉Python编程。[廖雪峰老师的教程](https://liaoxuefeng.com/books/python/introduction/index.html)提供了非常好的入门指引，建议学到 17.常用内建模块即可，而对7，8，9，10，11则要重点掌握。每一节学完以后尝试去完成课后习题。最后，**一定要以一个实践项目来检验自己的学习成果**，比如设计一段爬虫，或是实现一些小功能等 [(这是我写的一个小爬虫)](https://github.com/gtzjh/WundergroundSpider)。请不要在这一阶段使用ChatGPT，但可以再写出来以后让其给出代码优化建议让自己进步。

2. 机器学习的基础。吴恩达老师的[CS229](https://www.bilibili.com/video/BV1JE411w7Ub)课程是非常棒的资料。

3. 其他

    **明白如何使用conda和pip创建和管理环境**，并明白如何在编辑器（vscode等）中使用它

    **明白如何使用终端（Terminal）**

    建议学会使用[Git](https://github.com/gtzjh/learngit)，尝试自己在GitHub上建一个项目并学会用它来管理代码。

    建议使用VScode，

    *我个人不喜欢PyCharm，有太多花里胡哨的功能。更不建议使用Jupyter Notebook或Jupyterlab，因为很容易写出不够流畅优雅的代码。*



## 1. 环境准备（Windows平台，其余平台同理）

**使用 Python 3.10**

*环境安装大约使用1.75 GiB存储空间*

conda

```bash
conda env create -f env-win.yml -n mymodels
```

```bash
conda activate mymodels
```

## 2. 使用

根据自己需要修改 `main.py` 中的以下内容：

```python
file_path = "data.csv"              # 数据文件路径
y = "y"                             # 选择因变量（y）的列名
x_list = list(range(1, 16))         # 选择自变量（x）的列索引列表，也可以传入变量名字符串列表
model = "svr"                       # 模型选择："dt"、"rf"、"gbdt"、"ada"、"xgb"、"lgb"、"cat"、"svr"、"knr"、"mlp"
results_dir = "results/svr"         # 结果保存目录，可以使用模型名称作为目录名，也可以传入pathlib对象
cat_features = None                 # 分类特征列表，如 ["x16", "x17"]，没有则为None
trials = 50                         # Optuna超参数优化的尝试次数
test_ratio = 0.3                    # 测试集占总数据集的比例（占总数据集的30%）
shap_ratio = 0.3                    # 用于SHAP值计算的数据比例（占总数据集的30%）
cross_valid = 5                     # Optuna超参数优化时的交叉验证折数
random_state = 0                    # 全局随机种子，用于控制模型训练、交叉验证和测试的随机性
```

运行 main.py 。(命令行中或用Debug模式)

```bash
python main.py
```