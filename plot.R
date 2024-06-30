rm(list = ls())
setwd("C:/Users/jh/workspace/MyModels/")


library(tidyverse)
# library(showtext)
# showtext_opts(dpi = 500)
# showtext_auto()
# theme_set(theme_bw())
source("plot/Optimization.R")
source("plot/Scatter.R")


# Accuracy: training and test accuracy, scatter plots




# Parameters tuning using Optuna.
# optimization_plot <- plot_optimization("results/RF/optimization.csv")
# print(optimization_plot)

scatter_plot <- plot_scatter("results/RF/scatter_test.csv",
                             "results/RF/scatter_train.csv")

print(scatter_plot)