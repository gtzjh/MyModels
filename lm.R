# 多元线性回归
setwd("~/workspace/MyModels")
rm(list = ls())

library(tidyverse)

data <- read_csv("data/data.csv")

model <- lm(
  y ~ x1 + x2 + x3,
  data = data
)
print(summary(model))