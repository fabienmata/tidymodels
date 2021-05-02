
library(tidymodels) #containing rsample, recipe, parsnip, tune , yardstick
library(naniar) #NA handling
library(finalfit) #NA handling
library(workflowsets) 
library(glmnet) #for regularised logistic 
library(rpart) #for decision tree
#library(baguette) #for ensemble learning
library(randomForest) #self explaining
library(discrim) #for discriminant analysis 

risk <- read.csv("https://raw.githubusercontent.com/fabienmata/tidymodels/master/data/german_credit_data.csv", 
                 row.names = 'X',
                 stringsAsFactors = TRUE)

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
  
  #us the na's to create a new level 
  step_unknown(Saving.accounts, new_level = "no account") %>% 
  step_unknown(Checking.account, new_level = "no account") %>% 
  
  #normalize all numeric variables
  step_normalize(all_numeric()) %>% 
  
  #turn all the factors into dummies and delete the reference level
  step_dummy(all_nominal(), -all_outcomes()) 


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
                        v = 5,
                        #outcome variable
                        strata = Risk)

#model specification 

#tuned logit ==> regularised
logit_tune_model <- logistic_reg(penalty = tune(), 
                                 mixture = tune()) %>%
  set_engine('glmnet') %>%
  set_mode('classification')

#tuned discriminant analysis : a compromise between qda and lda by setting the hyperparameter penalty

rda_tuned <- discrim_regularized(frac_common_cov = tune(),
                                 frac_identity = tune()) %>% 
  set_engine('klaR') %>% 
  set_mode('classification')

#tuned decision tree
dt_tuned <- decision_tree(cost_complexity = tune(),
                               tree_depth = tune(),
                               min_n = tune()) %>%
  set_engine('rpart') %>%
  set_mode('classification')

#tuned random forest 
rf_tuned <- rand_forest(mtry = tune(),
                        trees = tune(),
                        min_n = tune()) %>% 
  set_engine('randomForest') %>%
  set_mode('classification')

knn_tuned <- nearest_neighbor(neighbors = tune(),
                              weight_func = tune(),
                              dist_power = tune()) %>% 
  set_engine('kknn') %>%
  set_mode('classification')

svm_rbf_tuned <- svm_rbf(cost = tune(),
                            rbf_sigma = tune()) %>% 
  set_engine('kernlab') %>% 
  set_mode('classification')



#turn the models into a list 
models <- list(#logit = logit_tune_model,
               #rda = rda_tune_model,
               #dt = dt_tune_model, 
               #rf = rf_tune_model,
               knn = knn_tuned,
               svm = svm_rbf_tuned)

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
   pull_workflow_set_result("rec_bagging") %>% 
   select_best(metric = "roc_auc")

#fit with the best model
final_fit <- risk_wflow_set %>% 
  pull_workflow("rec_bagging") %>% 
  finalize_workflow(best_results) %>% 
  last_fit(risk_split)

risk_results <- final_fit %>% collect_metrics()
risk_predictions <- final_fit %>% collect_predictions()

risk_results

conf_mat(risk_predictions,
         truth = Risk,
         estimate = .pred_class) %>%
  autoplot(type = 'heatmap')
