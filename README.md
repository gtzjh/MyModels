[English version](#English-Documentation)

[中文版本](#中文使用说明)


# On-going

# English Documentation

**Store my machine learning and SHAP (SHapley Additive exPlanations) codes.**

**DO REMEMBER: All models are wrong, but some are useful.**

*For regression task only currently.*

Models Supported:

1. [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)

2. [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

3. [LightGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html)

4. [Catboost](https://catboost.ai/en/docs/concepts/python-reference_catboostregressor)

5. [GBDT](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)

## 0. Former knowledge

1. Familiar with Python programming. [Liao Xuefeng's tutorial](https://liaoxuefeng.com/books/python/introduction/index.html) provides a very good guide to get started. It is recommended to learn **section 17**. Common built-in modules can be used, and **focus on section 7,8,9,10,11**. Try to complete the exercises after each lesson. Finally, you must test your learning results with a practical project [(Here is my demon)](https://github.com/gtzjh/WundergroundSpider), such as designing a crawler or implementing some small functions. Please do not use ChatGPT at this stage, but you can write it later and ask it to give suggestions for code optimization to improve yourself.

2. Fundamentals of machine learning. [CS229](https://www.bilibili.com/video/BV1JE411w7Ub) from Dr. Andrew Ng is a perfect tutorials.

3. Other
   
    Understand how to create and manage environments with conda and pip ** and understand how to use it in editors (VSCode, etc.)

    **Understand how to use Terminal**

    It is recommended to learn to use Git [Here is my little demon](https://github.com/gtzjh/learngit), try to build a project on GitHub yourself and learn to use it to manage the code.

    VScode is recommended.
    
    *Personally I don't recommend PyCharm, there are too many fancy features. It is also not recommended to use a Jupyter Notebook or Jupyterlab because it is easy to write less than smooth and elegant code.*


## 1. Prepare environment (On Windows platform)

**Please use Python 3.10**

*About 1.75 GiB storage would be used.*

conda

```bash
conda env create -f env-win.yml -n mymodels
```


```bash
conda activate mymodels
```

Or use the `pip3` when the `conda` command is not available.

```bash
pip install -r env-win.txt
```

## 2. Usage

Change the following content in `main.py` to meet your requirements.

```python
file_path = "data.csv"              # Where to load data
y = 0                               # Choose the index as dependency (y)，, you can also pass string of variables' name
x_list = list(range(2, 15))         # Choose the index as independency (x), you can also pass a list of string of variables' name
model = "lgb"                       # Model selection: "lgb", "cat", "rf", "dt", "gbdt".
results_dir = "results/"            # Use the model name as the results dir, you can also pass the pathlib object
trials = 100                        # How many trials to execute in optuna hyperparameters turning.
test_ratio = 0.3                    # Ratio for test in the whole dataset.
shap_ratio = 0.3                    # Use 30% of the whole dataset for SHAP calculation.
cross_valid = 5                     # Cross validation in optuna hyperparameters turning.
random_state = 0                    # Global random state control, for model training, cross validation turning, and testing.
```

Run main.py 。(In terminal or in Debug mode)

```bash
python main.py
```

************************************************************************************************************************************************************

# 中文使用说明

存储我常用的机器学习模型，并使用Optuna进行贝叶斯调参。用最少的时间完成机器学习任务。

**请记住：所有模型都是错的，但有一些是有用的。**

*目前仅支持回归任务。*

本项目包含的模型：

1. [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)

2. [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

3. [LightGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html)

4. [Catboost](https://catboost.ai/en/docs/concepts/python-reference_catboostregressor)

5. [GBDT](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)


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
conda env create -f env.yml -n mymodels
```

```bash
conda activate mymodels
```

如果无法使用conda，则使用pip

```bash
pip install -r win-env.txt
```

## 2. 使用

根据自己需要修改 `main.py` 中的以下内容：

```python
file_path = "data.csv"              # Where to load data
y = 0                               # Choose the index as dependency (y)，, you can also pass string of variables' name
x_list = list(range(2, 15))         # Choose the index as independency (x), you can also pass a list of string of variables' name
model = "lgb"                       # Model selection: "lgb", "cat", "rf", "dt", "gbdt".
results_dir = "results/"            # Use the model name as the results dir, you can also pass the pathlib object
trials = 100                        # How many trials to execute in optuna hyperparameters turning.
test_ratio = 0.3                    # Ratio for test in the whole dataset.
shap_ratio = 0.3                    # Use 30% of the whole dataset for SHAP calculation.
cross_valid = 5                     # Cross validation in optuna hyperparameters turning.
random_state = 0                    # Global random state control, for model training, cross validation turning, and testing.
```

运行 main.py 。(命令行中或用Debug模式)

```bash
python main.py
```