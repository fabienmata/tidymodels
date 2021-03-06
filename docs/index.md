Credit Risk with tidymodels
================
Fabien Mata
23/04/2021

## 1- The data

### Context

The dataset contains 1000 entries with 10 categorical/symbolic
attributes. In this dataset, each entry represents a person who takes a
credit by a bank. Each person is classified as good or bad credit risks
according to the set of attributes. The link to the original dataset can
be found
[here](https://www.kaggle.com/kabure/german-credit-data-with-risk?select=german_credit_data.csv).

### Content

Here are the variables :

1.  Age (numeric)

2.  Sex (text: male, female)

3.  Job (numeric: 0 - unskilled and non-resident, 1 - unskilled and
    resident, 2 - skilled, 3 - highly skilled)

4.  Housing (text: own, rent, or free)

5.  Saving accounts (text - little, moderate, quite rich, rich)

6.  Checking account (numeric, in DM - Deutsch Mark)

7.  Credit amount (numeric, in DM)

8.  Duration (numeric, in month)

9.  Purpose (text: car, furniture/equipment, radio/TV, domestic
    appliances, repairs, education, business, vacation/others)

10. Risk (text : good, bad)

## 2. Packages

The work was done using the tidymodels framework. is a collection of
packages for modeling and machine learning using tidyverse principles.
It is composed of the following packages :

-   ‘rsample’ and ‘recipes’ for preprocessing

-   ‘parsnip’ and ‘tune’ for modeling and model optimization. Parsnip
    depends on several machine learning engines, it then requires the
    presence of each package for each engine.

-   ‘yardstick’ for model validation

‘workfowsets’ is not part of the tidymodels collection of package, event
though it is a part of the **tidymodels** ecosystem. It helps to better
manage machine learning workflows.

``` r
library(tidymodels) 
library(naniar) #NA handling
library(finalfit) #NA handling
library(workflowsets) 
###engines for parsnip models 
library(glmnet) #for regularised logistic 
library(rpart) #for decision tree
library(randomForest) #self explaining
library(klaR) #for discriminant analysis (engine)
library(discrim) #for discriminant analysis (function)
library(kknn) #for nearest neighbor
library(kernlab) #for support vector machine
```

## 3. Exporatory analysis

Let’s take a first look at the data

``` r
risk <- read.csv("https://raw.githubusercontent.com/fabienmata/tidymodels/master/data/german_credit_data.csv", 
                 row.names = 'X',
                 stringsAsFactors = TRUE)
library(kableExtra)
```

    ## Warning: package 'kableExtra' was built under R version 4.0.5

    ## 
    ## Attaching package: 'kableExtra'

    ## The following object is masked from 'package:dplyr':
    ## 
    ##     group_rows

``` r
risk %>% head() %>% kbl() %>% kable_styling()
```

<table class="table" style="margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:left;">
</th>
<th style="text-align:right;">
Age
</th>
<th style="text-align:left;">
Sex
</th>
<th style="text-align:right;">
Job
</th>
<th style="text-align:left;">
Housing
</th>
<th style="text-align:left;">
Saving.accounts
</th>
<th style="text-align:left;">
Checking.account
</th>
<th style="text-align:right;">
Credit.amount
</th>
<th style="text-align:right;">
Duration
</th>
<th style="text-align:left;">
Purpose
</th>
<th style="text-align:left;">
Risk
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
0
</td>
<td style="text-align:right;">
67
</td>
<td style="text-align:left;">
male
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:left;">
own
</td>
<td style="text-align:left;">
NA
</td>
<td style="text-align:left;">
little
</td>
<td style="text-align:right;">
1169
</td>
<td style="text-align:right;">
6
</td>
<td style="text-align:left;">
radio/TV
</td>
<td style="text-align:left;">
good
</td>
</tr>
<tr>
<td style="text-align:left;">
1
</td>
<td style="text-align:right;">
22
</td>
<td style="text-align:left;">
female
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:left;">
own
</td>
<td style="text-align:left;">
little
</td>
<td style="text-align:left;">
moderate
</td>
<td style="text-align:right;">
5951
</td>
<td style="text-align:right;">
48
</td>
<td style="text-align:left;">
radio/TV
</td>
<td style="text-align:left;">
bad
</td>
</tr>
<tr>
<td style="text-align:left;">
2
</td>
<td style="text-align:right;">
49
</td>
<td style="text-align:left;">
male
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:left;">
own
</td>
<td style="text-align:left;">
little
</td>
<td style="text-align:left;">
NA
</td>
<td style="text-align:right;">
2096
</td>
<td style="text-align:right;">
12
</td>
<td style="text-align:left;">
education
</td>
<td style="text-align:left;">
good
</td>
</tr>
<tr>
<td style="text-align:left;">
3
</td>
<td style="text-align:right;">
45
</td>
<td style="text-align:left;">
male
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:left;">
free
</td>
<td style="text-align:left;">
little
</td>
<td style="text-align:left;">
little
</td>
<td style="text-align:right;">
7882
</td>
<td style="text-align:right;">
42
</td>
<td style="text-align:left;">
furniture/equipment
</td>
<td style="text-align:left;">
good
</td>
</tr>
<tr>
<td style="text-align:left;">
4
</td>
<td style="text-align:right;">
53
</td>
<td style="text-align:left;">
male
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:left;">
free
</td>
<td style="text-align:left;">
little
</td>
<td style="text-align:left;">
little
</td>
<td style="text-align:right;">
4870
</td>
<td style="text-align:right;">
24
</td>
<td style="text-align:left;">
car
</td>
<td style="text-align:left;">
bad
</td>
</tr>
<tr>
<td style="text-align:left;">
5
</td>
<td style="text-align:right;">
35
</td>
<td style="text-align:left;">
male
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:left;">
free
</td>
<td style="text-align:left;">
NA
</td>
<td style="text-align:left;">
NA
</td>
<td style="text-align:right;">
9055
</td>
<td style="text-align:right;">
36
</td>
<td style="text-align:left;">
education
</td>
<td style="text-align:left;">
good
</td>
</tr>
</tbody>
</table>

Check column info :

``` r
risk %>% str()
```

    ## 'data.frame':    1000 obs. of  10 variables:
    ##  $ Age             : int  67 22 49 45 53 35 53 35 61 28 ...
    ##  $ Sex             : Factor w/ 2 levels "female","male": 2 1 2 2 2 2 2 2 2 2 ...
    ##  $ Job             : int  2 2 1 2 2 1 2 3 1 3 ...
    ##  $ Housing         : Factor w/ 3 levels "free","own","rent": 2 2 2 1 1 1 2 3 2 2 ...
    ##  $ Saving.accounts : Factor w/ 4 levels "little","moderate",..: NA 1 1 1 1 NA 3 1 4 1 ...
    ##  $ Checking.account: Factor w/ 3 levels "little","moderate",..: 1 2 NA 1 1 NA NA 2 NA 2 ...
    ##  $ Credit.amount   : int  1169 5951 2096 7882 4870 9055 2835 6948 3059 5234 ...
    ##  $ Duration        : int  6 48 12 42 24 36 24 36 12 30 ...
    ##  $ Purpose         : Factor w/ 8 levels "business","car",..: 6 6 4 5 2 4 5 2 6 2 ...
    ##  $ Risk            : Factor w/ 2 levels "bad","good": 2 1 2 2 1 2 2 2 2 1 ...

### a. Missing values handling

Number of missing values per column :

``` r
risk %>% gg_miss_var()
```

![](rmd_files/figure-gfm/naniar-1.png)<!-- -->

Missing data points position :

``` r
risk %>% missing_plot()
```

![](rmd_files/figure-gfm/finalfit-1.png)<!-- -->

### b. Descriptive analysis

``` r
nums = c("Age", "Credit.amount", "Duration", "Job")
risk[nums] %>%                    
  gather() %>%                            
  ggplot(aes(value)) +        
  facet_wrap(~ key, scales = "free") +  
  geom_density()
```

![](rmd_files/figure-gfm/numerics-1.png)<!-- -->

``` r
require(corrplot)
```

    ## Loading required package: corrplot

    ## Warning: package 'corrplot' was built under R version 4.0.3

    ## corrplot 0.84 loaded

``` r
col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
corrplot(cor(risk[nums]), method = "color", col = col(200),  
         type = "upper", order = "hclust", 
         addCoef.col = "black", # Add coefficient of correlation
         tl.col = "darkblue", tl.srt = 45, #Text label color and rotation
         # Combine with significance level
         sig.level = 0.01,  
         # hide correlation coefficient on the principal diagonal
         diag = FALSE 
         )
```

![](rmd_files/figure-gfm/corrplot-1.png)<!-- -->

``` r
bg<-ggplot(risk, aes(x= Duration, y = Credit.amount)) +
  geom_point(aes(colour = Risk), alpha=.3) + 
  theme(axis.text.x=element_blank(), legend.position = "none") +facet_wrap(~Risk)
bg + labs(title = "Churners visit museum less", x= "", y="")
```

![](rmd_files/figure-gfm/credit%20amount%20vs%20duration-1.png)<!-- -->

``` r
ggplot(risk)+ 
  geom_density(aes(x = Credit.amount,fill = Risk), alpha=.5)+
  facet_wrap(~Saving.accounts)+
  labs(title = "Price pattern", y= "", x="Age")
```

![](rmd_files/figure-gfm/credit%20amount-1.png)<!-- -->

``` r
ggplot(data = risk,aes(x= Saving.accounts, y =Credit.amount, color = Risk)) + 
  geom_boxplot(size=1, alpha = .3) +
  scale_x_discrete() +
  scale_y_continuous()+ geom_jitter(aes(color=Risk), alpha=.2)
```

![](rmd_files/figure-gfm/saving%20accounts-1.png)<!-- -->

``` r
ggplot(data = risk,aes(x= Checking.account, y =Credit.amount, color = Risk)) + 
  geom_boxplot(size=1, alpha = .3) +
  scale_x_discrete() +
  scale_y_continuous()+ geom_jitter(aes(color=Risk), alpha=.2)
```

![](rmd_files/figure-gfm/checking%20account-1.png)<!-- -->

``` r
ggplot(data = risk,aes(x= Purpose, y =Credit.amount, color = Risk)) + 
  geom_boxplot(size=1, alpha = .3) +
  scale_x_discrete() +
  scale_y_continuous()+ geom_jitter(aes(color=Risk), alpha=.2)
```

![](rmd_files/figure-gfm/purpose-1.png)<!-- -->

## 4. Prepocessing

``` r
set.seed(1)
risk_split <- initial_split(risk,
                            prop = 0.75,
                            strata = Risk)

risk_training <- risk_split %>% 
  training()

risk_test <- risk_split %>% 
  testing()

#folds caracteristics for the cross validation 
set.seed(2)
risk_folds <- vfold_cv(data =  risk_training,
                       #number of partition
                       v = 3,
                       #outcome variable
                       strata = Risk)
```

``` r
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
```

``` r
logit_tuned <- logistic_reg(penalty = tune(), 
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
```

``` r
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
```

``` r
wflow_set_grid_results <- risk_wflow_set %>% 
  workflow_map(
  #tune_grid() parameters
    resamples = risk_folds,
    grid = 10,
    metrics = risk_metrics,
  #workflow_map() own parameters
    seed = 3,
    verbose = TRUE
)
```

    ## i 1 of 6 tuning:     rec_logit

    ## Warning: package 'rlang' was built under R version 4.0.5

    ## Warning: package 'vctrs' was built under R version 4.0.5

    ## v 1 of 6 tuning:     rec_logit (12.8s)

    ## i 2 of 6 tuning:     rec_rda

    ## v 2 of 6 tuning:     rec_rda (15.2s)

    ## i 3 of 6 tuning:     rec_dt

    ## v 3 of 6 tuning:     rec_dt (13.5s)

    ## i 4 of 6 tuning:     rec_rf

    ## i Creating pre-processing data to finalize unknown parameter: mtry

    ## v 4 of 6 tuning:     rec_rf (39.2s)

    ## i 5 of 6 tuning:     rec_knn

    ## v 5 of 6 tuning:     rec_knn (26.8s)

    ## i 6 of 6 tuning:     rec_svm

    ## v 6 of 6 tuning:     rec_svm (15.9s)

``` r
#rank the models by the area under the roc curve
wflow_set_grid_results %>% 
  rank_results(rank_metric = "roc_auc") %>% 
  filter(.metric == "roc_auc")
```

    ## # A tibble: 60 x 9
    ##    wflow_id  .config     .metric  mean std_err     n preprocessor model     rank
    ##    <chr>     <chr>       <chr>   <dbl>   <dbl> <int> <chr>        <chr>    <int>
    ##  1 rec_rda   Preprocess~ roc_auc 0.748  0.0249     3 recipe       discrim~     1
    ##  2 rec_rf    Preprocess~ roc_auc 0.748  0.0218     3 recipe       rand_fo~     2
    ##  3 rec_rda   Preprocess~ roc_auc 0.747  0.0257     3 recipe       discrim~     3
    ##  4 rec_rf    Preprocess~ roc_auc 0.747  0.0208     3 recipe       rand_fo~     4
    ##  5 rec_rf    Preprocess~ roc_auc 0.745  0.0169     3 recipe       rand_fo~     5
    ##  6 rec_rf    Preprocess~ roc_auc 0.744  0.0228     3 recipe       rand_fo~     6
    ##  7 rec_rf    Preprocess~ roc_auc 0.744  0.0153     3 recipe       rand_fo~     7
    ##  8 rec_logit Preprocess~ roc_auc 0.743  0.0245     3 recipe       logisti~     8
    ##  9 rec_logit Preprocess~ roc_auc 0.742  0.0249     3 recipe       logisti~     9
    ## 10 rec_rf    Preprocess~ roc_auc 0.742  0.0187     3 recipe       rand_fo~    10
    ## # ... with 50 more rows

``` r
#plot the performance of each model by rank
wflow_set_grid_results %>% 
  autoplot(rank_metric= "roc_auc", 
           metric = "roc_auc")
```

![](rmd_files/figure-gfm/rank%20plot-1.png)<!-- -->

``` r
#take the best result
best_results <- wflow_set_grid_results %>% 
  pull_workflow_set_result("rec_rf") %>% 
  select_best(metric = "roc_auc")

#fit the best model
final_fit <- wflow_set_grid_results %>% 
  pull_workflow("rec_rf") %>% 
  finalize_workflow(best_results) %>% 
  last_fit(risk_split)
```

``` r
final_fit %>% collect_metrics()
```

    ## # A tibble: 2 x 4
    ##   .metric  .estimator .estimate .config             
    ##   <chr>    <chr>          <dbl> <chr>               
    ## 1 accuracy binary         0.732 Preprocessor1_Model1
    ## 2 roc_auc  binary         0.767 Preprocessor1_Model1

``` r
risk_predictions <- final_fit %>% collect_predictions()
conf_mat(risk_predictions,
         truth = Risk,
         estimate = .pred_class) %>%
  autoplot(type = 'heatmap')
```

![](rmd_files/figure-gfm/heatmap-1.png)<!-- -->
