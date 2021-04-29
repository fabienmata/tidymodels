
library(rsample)
library(recipes)
library(parsnip)
library(yardstick)
library(tune)
library(naniar)
library(finalfit)

risk <- read.csv("https://raw.githubusercontent.com/fabienmata/tidymodels/master/data/german_credit_data.csv", row.names = 'X')

View(risk)
#which variable has missing values
risk %>% gg_miss_var()
#see clearly the missing data points
risk %>% missing_plot()

#split the dataset
risk_split <- initial_split(risk,
                            prop = 0.75,
                            strata = Risk)

risk_training <- risk_split %>% 
  training()

risk_test <- risk_split %>% 
  testing()

### need to find a way to turn na's to 0 
risk_rec <- recipe(Risk ~., data = risk_training) %>% 
  
  step_string2factor(all_nominal()) %>% 
  step_unknown(Saving.accounts, new_level = "no account") %>% 
  step_unknown(Checking.account, new_level = "no account")

risk_rec_prep <- risk_rec %>% 
  prep(training= risk_training)

risk_rec_prep %>% 
  bake(new_data = NULL)
