
library(tidymodels) #containing rsample, recipe, parsnip, tune , yardstick
library(naniar)
library(finalfit)
library(workflowsets)
library(glmnet)
library(rpart)

risk <- read.csv("https://raw.githubusercontent.com/fabienmata/tidymodels/master/data/german_credit_data.csv", 
                 row.names = 'X')

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
  step_string2factor(all_nominal(), -all_outcomes()) %>%
  
  #us the na's to create a new level 
  step_unknown(Saving.accounts, new_level = "no account") %>% 
  step_unknown(Checking.account, new_level = "no account") %>% 
  
  #normalize all numeric variables
  step_normalize(all_numeric()) %>% 
  
  #turn all the factors into dummies and delete the reference level
  step_dummy(all_nominal(), -all_outcomes()) 

#prep the recipe 
risk_rec_prep <- risk_rec %>% 
  prep(training= risk_training)

risk_rec_prep %>% 
  bake(new_data = NULL)

#training data
risk_training_prep <- risk_rec_prep %>% 
  bake(new_data = NULL)

#testing data
risk_test_prep <- risk_rec_prep %>% 
  bake(new_data = risk_test)

# just to check if there are any correlation between the predictor
# there are not so no need to add corr to our recipe object
# z <- risk_baked %>%
#   select_if(is.numeric) %>%
#   cor()
# zdf <- as.data.frame(as.table(z))
# subset(zdf, abs(Freq) > 0.8)

#model specification
#logistic regression

#metrics we want for each model 
#we want : accuracy, sensitivity, specificity, area under the roc curve 
risk_metrics <- metric_set(accuracy, sens, spec, roc_auc)

#folds caracteristics for the cross validation 
risk_folds <- vfold_cv(data =  risk_training,
                        #nb of folds
                        v = 10,
                        #outcome variable
                        strata = Risk)

#model specification 

#tuned logit 
logit_tune_model <- logistic_reg(penalty = tune(), 
                                 mixture = tune()) %>%
  set_engine('glmnet') %>%
  set_mode('classification')

#tuned decision tree
dt_tune_model <- decision_tree(cost_complexity = tune(),
                               tree_depth = tune(),
                               min_n = tune()) %>%
  set_engine('rpart') %>%
  set_mode('classification')


#turn the models into a list 
models <- list(dt = dt_tune_model, logit = logit_tune_model)

#incorporate them in a set of workflow
risk_wflow_set <- workflow_set(list(rec = risk_rec), models, cross = TRUE)  

#tuning the model <- cross validation
risk_wflow_set <- risk_wflow_set %>% 
  workflow_map(
    
  # Options to `tune_grid()
    resamples = risk_folds,
    grid = 10,
    metrics = risk_metrics,
  # Options to `workflow_map()`
    seed = 3,
    verbose = TRUE
)

#rank the models by the area under the roc curve
risk_wflow_set %>% 
  rank_results(rank_metric = "roc_auc") %>% 
  filter(.metric == "roc_auc")

risk_wflow_set %>% 
  autoplot(rank_metric= "roc_auc", metric = "roc_auc")


# predict ----------------------------------------------------------------------
#take the best result
best_results <- risk_wflow_set %>% 
   pull_workflow_set_result("rec_logit") %>% 
   select_best(metric = "roc_auc")

#fit with the best model
final_fit <- risk_wflow_set %>% 
  pull_workflow("rec_logit") %>% 
  finalize_workflow(best_results) %>% 
  fit(risk_training)

class_preds <- final_fit %>%
  predict(new_data = risk_test,
          type = 'class')

prob_preds <- final_fit %>%
  predict(new_data = risk_test,
          type = 'prob')

risk_results <- risk_test %>%
  select(Risk) %>%
  bind_cols(class_preds, prob_preds)

risk_results

conf_mat(risk_results,
         truth = Risk,
         estimate = .pred_class) %>%
  autoplot(type = 'heatmap')
