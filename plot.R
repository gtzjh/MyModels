rm(list = ls())
setwd("C:/Users/jh/workspace/MyModels/")


library(tidyverse)
# library(showtext)
# showtext_opts(dpi = 500)
# showtext_auto()
theme_set(theme_bw())

source("plot/Optimization.R")
source("plot/Scatter.R")
source("plot/GlobalExplanation.R")
source("plot/LocalExplanation.R")


# Accuracy: training and test accuracy, scatter plots
# scatter_plot <- plot_scatter("results/RF/scatter_test.csv",
#                              "results/RF/scatter_train.csv",
#                              "results/RF/accuracy.yml")
# print(scatter_plot)


# Parameters tuning using Optuna.
# optimization_plot <- plot_optimization("results/RF/optimization.csv")
# print(optimization_plot)


# Global explanation
# global_explanation_plot <- plot_global("results/RF/shap_values.csv")
# print(global_explanation_plot)


# Local explanation
# local_explanation_plot <- plot_local("results/RF/shap_data.csv",
#                                      "results/RF/shap_values.csv")
# print(local_explanation_plot)




