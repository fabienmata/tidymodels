
library(tidymodels) #containing rsample, recipe, parsnip, tune , yardstick
library(naniar)
library(finalfit)

risk <- read.csv("https://raw.githubusercontent.com/fabienmata/tidymodels/master/data/german_credit_data.csv", row.names = 'X')

#look at the data as is
risk %>% head()

risk %>% str()

#which variable has missing values
#from the naniar package
risk %>% gg_miss_var()
#see clearly the missing data points
#from the finalfit package
risk %>% missing_plot()

#split the dataset
#using the rsample package
risk_split <- initial_split(risk,
                            prop = 0.75,
                            strata = Risk)

risk_training <- risk_split %>% 
  training()

risk_test <- risk_split %>% 
  testing()

### need to find a way to turn na's to 0 
#using recipe package
risk_rec <- recipe(Risk ~., data = risk_training) %>% 
  
  #turn nominal variables into factors
  step_string2factor(all_nominal()) %>%
  
  #us the na's to create a new level 
  step_unknown(Saving.accounts, new_level = "no account") %>% 
  step_unknown(Checking.account, new_level = "no account") %>% 
  
  #normalize all numeric variables
  step_normalize(all_numeric()) %>% 
  
  #turn all the factors into dummies and delete the reference level
  step_dummy(all_nominal(), -all_outcomes())

#
risk_rec_prep <- risk_rec %>% 
  prep(training= risk_training)

risk_rec_prep %>% 
  bake(new_data = NULL)

risk_baked <- risk_rec_prep %>% 
  bake(new_data = NULL)

# just to check if there are any correlation between the predictor
#there are not so no need to add corr to our recipe object
'''z <- risk_baked %>%
  select_if(is.numeric) %>%
  cor()
zdf <- as.data.frame(as.table(z))
subset(zdf, abs(Freq) > 0.8)
'''
# 
