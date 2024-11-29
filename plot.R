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
# source("plot/GlobalExplanation.R")
# source("plot/LocalExplanation.R")

setwd("results/gbdt_3/")  # Navigate to the results dir

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


# Global explanation
# global_explanation_plot <- plot_global("shap_values.csv")
# print(global_explanation_plot)
# ggsave("global_explanation_plot.png", plot = global_explanation_plot,
#        width = 140, height = 190, units = "mm")
# ggsave("global_explanation_plot.pdf", plot = global_explanation_plot,
#        width = 140, height = 190, units = "mm")


# Local explanation
# The local explanation may be very large, adjut the output width and height as you need. Here is in a A4 size.
# local_explanation_plot <- plot_local("shap_data.csv", "shap_values.csv")
# print(local_explanation_plot)
# ggsave("local_explanation_plot.png", plot = local_explanation_plot,
#        width = 190, height = 210, units = "mm")
# ggsave("local_explanation_plot.pdf", plot = local_explanation_plot,
#        width = 190, height = 210, units = "mm")
