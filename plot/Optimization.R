# Plot the optimization curve usint Optuna
rm(list = ls())

library(tidyverse)
# library(showtext)
# showtext_opts(dpi = 500)
# showtext_auto()
# theme_set(theme_bw())


plot_optimization <- function(optimization_file_path){
  optimization <- read_csv(optimization_file_path, show_col_types = FALSE)

  optimization_plot <- optimization %>%
    ggplot(aes(x = trials)) +
    geom_point(aes(y = current_accuracy, color = "Current Accuracy"), size = 4, alpha = 0.5) +
    geom_line(aes(y = best_accuracy, color = "Best Accuracy"), linewidth = 0.6) +
    labs(x = "Trials", y = "Accuracy (R2)", color = "") + 
    scale_color_manual(values = c("Current Accuracy" = "#4682B4", "Best Accuracy" = "#CD5C5C")) +
    theme(legend.position = "bottom",
          text = element_text(size = 12, family = "serif"))
  
  return(optimization_plot)
}
  
