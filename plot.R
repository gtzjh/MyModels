rm(list = ls())
setwd("C:/Users/jh/workspace/MyModels/")

library(tidyverse)
# library(ggExtra)  # For plotting the marginal histogram in scatter plot.
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
scatter_plot <- plot_scatter("results/CAT/scatter_test.csv",
                             "results/CAT/scatter_train.csv",
                             "results/CAT/accuracy.yml")
print(scatter_plot)
ggsave("results/CAT/scatter_plot.png", plot = scatter_plot,
       width = 140, height = 140, units = "mm")


# Parameters tuning using Optuna.
optimization_plot <- plot_optimization("results/CAT/optimization.csv")
print(optimization_plot)
ggsave("results/CAT/optimization_plot.png", plot = optimization_plot,
       width = 140, height = 140, units = "mm")


# Global explanation
global_explanation_plot <- plot_global("results/CAT/shap_values.csv")
print(global_explanation_plot)
ggsave("results/CAT/global_explanation_plot.png", plot = global_explanation_plot,
       width = 190, height = 190, units = "mm")


# Local explanation
local_explanation_plot <- plot_local("results/CAT/shap_data.csv",
                                     "results/CAT/shap_values.csv")
print(local_explanation_plot)
ggsave("results/CAT/local_explanation_plot.png", plot = local_explanation_plot,
       width = 210, height = 297, units = "mm")

