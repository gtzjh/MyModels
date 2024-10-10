rm(list = ls())
setwd("C:/Users/jh/workspace/MyModels/")
results_dir <- "results/CAT/"

library(tidyverse)
library(yaml)
library(patchwork)
library(showtext)
showtext_opts(dpi = 500)
showtext_auto()
theme_set(theme_bw())

source("plot/Scatter.R")
source("plot/Optimization.R")
source("plot/GlobalExplanation.R")
source("plot/LocalExplanation.R")


# Accuracy: training and test accuracy, scatter plots
scatter_plot <- plot_scatter(paste(results_dir, "scatter_test.csv", sep = ""),
                             paste(results_dir, "scatter_train.csv", sep = ""),
                             paste(results_dir, "accuracy.yml", sep = ""))
print(scatter_plot)
ggsave(paste(results_dir, "scatter_plot.png", sep = ""), plot = scatter_plot,
       width = 140, height = 140, units = "mm")


# Parameters tuning using Optuna.
optimization_plot <- plot_optimization(paste(results_dir, "optimization.csv", sep = ""))
print(optimization_plot)
ggsave(paste(results_dir, "optimization_plot.png", sep = ""), plot = optimization_plot,
       width = 140, height = 140, units = "mm")


# Global explanation
global_explanation_plot <- plot_global(paste(results_dir, "shap_values.csv", sep = ""))
print(global_explanation_plot)
ggsave(paste(results_dir, "global_explanation_plot.png", sep = ""), plot = global_explanation_plot,
       width = 190, height = 190, units = "mm")


# Local explanation
local_explanation_plot <- plot_local(paste(results_dir, "shap_data.csv", sep = ""),
                                     paste(results_dir, "shap_values.csv", sep = ""))
print(local_explanation_plot)
ggsave(paste(results_dir, "local_explanation_plot.png", sep = ""), plot = local_explanation_plot,
       width = 210, height = 297, units = "mm")

