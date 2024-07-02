plot_global <- function(global_explanation_path, color = "#6295A2"){
  # Read files and trans it to long dataframe.
  global_explanation <- read_csv(global_explanation_path, show_col_types = FALSE) %>%
    pivot_longer(cols = everything(), names_to = "factors", values_to = "shap_values")
  
  # Reorder the factors by their mean absolute values.
  f <- function(x){mean(abs(x))}
  abs_mean_shap_values <- aggregate(shap_values ~ factors,
                                    data = global_explanation,
                                    FUN = f)
  global_explanation$factors <- global_explanation$factors %>%
    factor(levels = abs_mean_shap_values$factors[order(abs_mean_shap_values$shap_values,
                                                       decreasing = FALSE)])
  
  global_explanation_plot <- global_explanation %>%
    ggplot(aes(x = shap_values, y = factors)) +
    geom_boxplot(size = 0.6, color = color) +
    labs(x = "", y = "") +
    theme(text = element_text(size = 12, family = "serif"))
  
  
  return(global_explanation_plot)
}
