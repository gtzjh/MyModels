rm(list = ls())

library(tidyverse)


group_color <- c("test" = "#CD5C5C", "train" = "#4682B4")
annotate_text <- "
  Test:
    R2      :   0.70
    RMSE: 89.66
    MAE  : 45.36
  Train:
    R2      :   0.80
    RMSE: 65.96
    MAE  : 35.08
"

plot_scatter <- function(test_data_path, train_data_path){
  # ----------------------------------------------------------------------------
  scatter_test <- read_csv(test_data_path, show_col_types = FALSE) %>%
    rename("actual" = "y_test", "pred" = "y_pred")
  scatter_test$group <- "test"
  
  scatter_train <- read_csv(train_data_path, show_col_types = FALSE) %>%
    rename("actual" = "y_train", "pred" = "y_train_pred")
  scatter_train$group <- "train"
  scatter_data <- bind_rows(list(scatter_train, scatter_test))
  # ----------------------------------------------------------------------------
  
  # ----------------------------------------------------------------------------
  scatter_plot <- scatter_data %>%
    ggplot(aes(x = actual, y = pred, group = group, color = group)) +
    geom_abline(intercept = 0, slope = 1, color = "gray") +
    geom_point(alpha = 0.3, size = 5) +
    scale_color_manual(values = group_color) +
    # xlim(1, 1600) + ylim(1, 1600) + 
    annotate("label",
             x = 50, y = 1600,
             label = annotate_text, 
             hjust = 0, vjust = 1, 
             size = 4, family = "serif", fill = "#6295A2", alpha = 0.1) +
    theme(aspect.ratio = 1, 
          legend.position = "bottom",
          text = element_text(size = 12, family = "serif"))
  # ----------------------------------------------------------------------------
  
  return(scatter_plot)
}


