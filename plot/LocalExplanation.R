plot_local <- function(shap_data_path, shap_values_path){
  #-----------------------------------------------------------------------------
  shap_data <- read_csv(shap_data_path, show_col_types = FALSE)
  shap_values <- read_csv(shap_values_path, show_col_types = FALSE)
  #-----------------------------------------------------------------------------
  
  #-----------------------------------------------------------------------------
  # Iter all factors
  factors_list <- colnames(shap_data)
  plot_list <- list()
  ind <- 1
  for (i in factors_list){
    # Extract one factor data for plotting.
    fig <- data.frame(x = shap_data[[i]], y = shap_values[[i]]) %>%
      ggplot(aes(x = x, y = y)) + 
      geom_point(size = 1.2, color = "#6295A2", alpha = 0.25) +
      geom_smooth(method = "lm", formula = y ~ poly(x, 2), se = FALSE, linewidth = 0.3) + 
      labs(title = i, x = "", y = "") +
      theme(text = element_text(size = 12, family = "serif"))

    plot_list[[ind]] <- fig
    ind <- ind + 1
  }
  #-----------------------------------------------------------------------------
  
  local_explanation_plot <- wrap_plots(plot_list) + plot_layout(ncol = 3)
  return(local_explanation_plot)
}