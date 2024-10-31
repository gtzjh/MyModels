group_color <- c("test" = "#CD5C5C", "train" = "#4682B4")


plot_scatter <- function(test_data_path, train_data_path, accuracy_path){
  # ----------------------------------------------------------------------------
  model_performance <- yaml.load_file(accuracy_path)
  test_r2 <- model_performance$test_accuracy$R2
  test_rmse <- model_performance$test_accuracy$RMSE
  test_mae <- model_performance$test_accuracy$MAE
  train_r2 <- model_performance$train_accuracy$R2
  train_rmse <- model_performance$train_accuracy$RMSE
  train_mae <- model_performance$train_accuracy$MAE

  
  annotate_text <- sprintf("
    Test:
      R2      :   %.3f
      RMSE: %.3f
      MAE  : %.3f
    Train:
      R2      :   %.3f
      RMSE: %.3f
      MAE  : %.3f
  ", test_r2, test_rmse, test_mae, train_r2, train_rmse, train_mae)
  # ----------------------------------------------------------------------------
  
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
    geom_point(alpha = 0.2, size = 3) +
    scale_color_manual(values = group_color) +
    # annotate("label",
    #          x = max(scatter_data$actual), y = max(scatter_data$pred),
    #          label = annotate_text, 
    #          hjust = 0, vjust = 1, 
    #          size = 2.5, family = "serif", fill = "#6295A2", alpha = 0.1)
    theme(aspect.ratio = 1,
          legend.position = "bottom",
          text = element_text(size = 12, family = "serif"))
  # ----------------------------------------------------------------------------
  return(scatter_plot)
}


