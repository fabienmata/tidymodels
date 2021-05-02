
library(tidymodels) #containing rsample, recipe, parsnip, tune , yardstick
library(naniar) #NA handling
library(finalfit) #NA handling
library(workflowsets) 
###engines for parsnip models 
library(glmnet) #for regularised logistic 
library(rpart) #for decision tree
library(randomForest) #self explaining
library(discrim)
library(klaR) #for discriminant analysis 
library(kknn) #for nearest neighbor
library(kernlab) #for support vector machine

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
  #set the event/reference level to 'good'
  step_relevel(Risk, ref_level = 'good') %>% 
  #us the na's to create a new level 
  step_unknown(Saving.accounts, new_level = "no account") %>% 
  step_unknown(Checking.account, new_level = "no account") %>% 
  
  #normalize all numeric variables
  step_normalize(all_numeric()) %>% 
  
  #turn all the factors into dummies and delete the reference level
  step_dummy(all_nominal(), -all_outcomes()) 

#model specification

#regularized logit 
logit_tune_model <- logistic_reg(penalty = tune(), 
                                 mixture = tune()) %>%
  set_engine('glmnet') %>%
  set_mode('classification')

#regularised discriminant analysis : a compromise between qda and lda by setting the hyperparameter penalty

rda_tuned <- discrim_regularized(frac_common_cov = tune(),
                                 frac_identity = tune()) %>% 
  set_engine('klaR') %>% 
  set_mode('classification')

#decision tree
dt_tuned <- decision_tree(cost_complexity = tune(),
                               tree_depth = tune(),
                               min_n = tune()) %>%
  set_engine('rpart') %>%
  set_mode('classification')

#random forest 
rf_tuned <- rand_forest(mtry = tune(),
                        trees = tune(),
                        min_n = tune()) %>% 
  set_engine('randomForest') %>%
  set_mode('classification')

#k nearest neighbors
knn_tuned <- nearest_neighbor(neighbors = tune(),
                              weight_func = tune(),
                              dist_power = tune()) %>% 
  set_engine('kknn') %>%
  set_mode('classification')

#support vector machine
svm_poly_tuned <- svm_poly(cost = tune(),
                          degree = tune(),
                          scale_factor = tune()) %>% 
  set_engine('kernlab') %>% 
  set_mode('classification')



#make a list out of the models
models <- list(logit = logit_tuned,
               rda = rda_tuned,
               dt = dt_tuned, 
               rf = rf_tuned,
               knn = knn_tuned,
               svm = svm_poly_tuned)

#incorporate them in a set of workflow
risk_wflow_set <- workflow_set(preproc = list(rec = risk_rec), 
                               models = models, 
                               cross = TRUE)  

#metrics we want for each model 
#we want : accuracy, sensitivity, specificity, area under the roc curve 
risk_metrics <- metric_set(accuracy, sens, spec, roc_auc)

#folds caracteristics for the cross validation 
risk_folds <- vfold_cv(data =  risk_training,
                       #nb of folds
                       v = 5,
                       #outcome variable
                       strata = Risk)

#tuning the model <- cross validation
wflow_set_grid_results <- risk_wflow_set %>% 
  workflow_map(
    
  #tune_grid() parameters
    resamples = risk_folds,
    grid = 10,
    metrics = risk_metrics,
  # workflow_map() own parameters
    seed = 3,
    verbose = TRUE
)

#rank the models by the area under the roc curve
wflow_set_grid_results %>% 
  rank_results(rank_metric = "roc_auc") %>% 
  filter(.metric == "roc_auc")

#plot the performance of each model by rank
wflow_set_grid_results %>% 
  autoplot(rank_metric= "roc_auc", 
           metric = "roc_auc")


# predict ----------------------------------------------------------------------
#take the best result
best_results <- wflow_set_grid_results %>% 
   pull_workflow_set_result("rec_svm") %>% 
   select_best(metric = "roc_auc")

#fit the best model
final_fit <- wflow_set_grid_results %>% 
  pull_workflow("rec_svm") %>% 
  finalize_workflow(best_results) %>% 
  last_fit(risk_split)

#result metrics of the best model
final_fit %>% collect_metrics()

#confusion matrix of the best model
risk_predictions <- final_fit %>% collect_predictions()
conf_mat(risk_predictions,
         truth = Risk,
         estimate = .pred_class) %>%
  autoplot(type = 'heatmap')
