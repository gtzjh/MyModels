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

## 0. Former knowledge

1. Familiar with Python programming. Liao Xuefeng's tutorial provides a very good guide to get started. It is recommended to learn **section 17**. Common built-in modules can be used, and **focus on section 7,8,9,10,11**. Try to complete the exercises after each lesson. Finally, you must test your learning results with a practical project, such as designing a crawler or implementing some small functions. Please do not use ChatGPT at this stage, but you can write it later and ask it to give suggestions for code optimization to improve yourself.

    > https://liaoxuefeng.com/books/python/introduction/index.html

2. Fundamentals of machine learning. CS229 from Dr. Andrew Ng is a perfect tutorials.

    > https://www.bilibili.com/video/BV1JE411w7Ub

3. Other
   
    Understand how to create and manage environments with conda and pip ** and understand how to use it in editors (VSCode, etc.)

    **Understand how to use Terminal**

    It is recommended to learn to use Git, try to build a project on GitHub yourself and learn to use it to manage the code.

    VScode is recommended.
    
    *Personally I don't recommend PyCharm, there are too many fancy features. It is also not recommended to use a Jupyter Notebook or Jupyterlab because it is easy to write less than smooth and elegant code.*


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

## 2. Usage

### 2.1 Model Training and Verification

Execute the main program file `main.py`.

Alter lines 8 - 13 of `main.py` to meet your requirements.

```python
model = "cat"                # Model selection: "lgb", "cat", "rf", "dt", respectively representing LightGBM, CatBoost, Random Forest, Decision Tree.
results_dir = "results/CAT"  # The folder where all results are saved. By default, a folder with the same name as the model abbreviation is created under the'results' folder.
trials = 50                  # The number of Trials performed using the Optuna tuning parameter. The more Trials are conducted, the more hyperparameter possibilities will be traversed. The default value is 50. You do not need to modify it if not necessary.
shap_ratio = 0.1             # The proportion of samples used to calculate the SHAP value. The default is 10%. Increasing this value will increase the running time but yield relatively more accurate results, and vice versa.
cross_valid = 6              # [No modification] The number of times each Trial is cross-verified using Optuna.
random_state = 6             # The global random seed for model training, cross-validation turning, and testing.
```

### 2.2 SHAP Interpretation and Plotting

Currently, only global interpretation and local interpretation are supported.

Install R (version 4.4.0 and above) and RStudio.

Open the MyModels.Rproj project file.

Open the plot.R plotting program.

1. Set the working directory, that is, `setwd()`, and modify it according to your local path. It should point to the path of the `MyModels` folder.

2. `results_dir <-` Please note that you must save the results in the folder you specify for yourself, corresponding to'results_dir' in the above main program.

Verify that all parameters are correct. Generally, only the above two parameters need to be modified.

Run the script.

Result Interpretation: In the directory corresponding to `results_dir`, you can see pictures with the `.png` suffix.

`scatter_plot.png` represents the model training and testing accuracy.

`optimization_plot.png` shows the model validation accuracy for each Trial in the tuning process and the optimal accuracy up to the current Trial.

`global_explanation_plot.png` is the global explanation, sorted from top to bottom by $mean(|SHAP|)$, representing the importance of the variable.

`local_explanation_plot.png` is the local interpretation, which can be used to explore the influence thresholds and inflection points of each variable.


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


## 0. 前置知识

1. 熟悉Python编程。廖雪峰老师的教程提供了非常好的入门指引，建议学到 17.常用内建模块即可，而对7，8，9，10，11则要重点掌握。每一节学完以后尝试去完成课后习题。最后，**一定要以一个实践项目来检验自己的学习成果**，比如设计一段爬虫，或是实现一些小功能等。请不要在这一阶段使用ChatGPT，但可以再写出来以后让其给出代码优化建议让自己进步。

    > https://liaoxuefeng.com/books/python/introduction/index.html

2. 机器学习的基础。吴恩达老师的CS229课程是非常棒的资料。

    > https://www.bilibili.com/video/BV1JE411w7Ub

3. 其他

    **明白如何使用conda和pip创建和管理环境**，并明白如何在编辑器（vscode等）中使用它

    **明白如何使用终端（Terminal）**

    建议学会使用Git，尝试自己在GitHub上建一个项目并学会用它来管理代码。

    建议使用VScode，

    *我个人不喜欢PyCharm，有太多花里胡哨的功能。更不建议使用Jupyter Notebook或Jupyterlab，因为很容易写出不够流畅优雅的代码。*



## 1. 环境准备（Windows平台，其余平台同理）

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

## 2. 使用

### 2.1 模型训练及验证

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

### 2.2 SHAP解释绘图

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
