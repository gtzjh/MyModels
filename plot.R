rm(list = ls())
setwd("C:/Users/jh/workspace/MyModels/")  # To the results dir

library(tidyverse)
library(ggbeeswarm)
library(yaml)
library(patchwork)
library(showtext)
showtext_opts(dpi = 300)
showtext_auto()
theme_set(theme_bw())

source("plot/Scatter.R")
source("plot/Optimization.R")

setwd("results/rf_5/")  # Navigate to the results dir

# Accuracy: training and test accuracy, scatter plots
scatter_plot <- plot_scatter("scatter_test.csv", "scatter_train.csv", "accuracy.yml")
print(scatter_plot)
ggsave("scatter_plot.png", plot = scatter_plot,
       width = 90, height = 90, units = "mm")
ggsave("scatter_plot.pdf", plot = scatter_plot,
       width = 90, height = 90, units = "mm")


# Parameters tuning using Optuna.
optimization_plot <- plot_optimization("optimization.csv")
print(optimization_plot)
ggsave("optimization_plot.png", plot = optimization_plot,
       width = 90, height = 90, units = "mm")
ggsave("optimization_plot.pdf", plot = optimization_plot,
       width = 90, height = 90, units = "mm")