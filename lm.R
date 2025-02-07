# 多元线性回归
setwd("~/workspace/MyModels")
rm(list = ls())

library(tidyverse)

data <- read_csv("data.csv")

model <- lm(
  y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 ,
  data = data
)
print(summary(model))