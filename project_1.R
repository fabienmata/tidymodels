
library(tidyverse)
library(rsample)
library(recipe)
library(parsnip)
library(yardstick)
library(tune)

risk <- read.csv("https://raw.githubusercontent.com/fabienmata/tidymodels/master/data/german_credit_data.csv")

