Credit Risk with tidymodels
================
Fabien Mata
23/04/2021

### Load tidymodels packages

``` r
library(tidymodels) #containing rsample, recipe, parsnip, tune , yardstick
library(naniar) #NA handling
library(finalfit) #NA handling
library(workflowsets) 
###engines for parsnip models 
library(glmnet) #for regularised logistic 
library(rpart) #for decision tree
library(randomForest) #self explaining
library(klaR) #for discriminant analysis 
library(kknn) #for nearest neighbor
library(kernlab) #for support vector machine
```

The data is from kaggle (‘german credit risk’) It contains 10 variables,
where

``` r
risk <- read.csv("https://raw.githubusercontent.com/fabienmata/tidymodels/master/data/german_credit_data.csv", 
                 row.names = 'X',
                 stringsAsFactors = TRUE)
risk %>% head()
```

    ##   Age    Sex Job Housing Saving.accounts Checking.account Credit.amount
    ## 0  67   male   2     own            <NA>           little          1169
    ## 1  22 female   2     own          little         moderate          5951
    ## 2  49   male   1     own          little             <NA>          2096
    ## 3  45   male   2    free          little           little          7882
    ## 4  53   male   2    free          little           little          4870
    ## 5  35   male   1    free            <NA>             <NA>          9055
    ##   Duration             Purpose Risk
    ## 0        6            radio/TV good
    ## 1       48            radio/TV  bad
    ## 2       12           education good
    ## 3       42 furniture/equipment good
    ## 4       24                 car  bad
    ## 5       36           education good

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

``` r
risk %>% gg_miss_var()
```

![](rmd_files/figure-gfm/naniar-1.png)<!-- -->

see clearly the missing data points

``` r
risk %>% missing_plot()
```

![](rmd_files/figure-gfm/finalfit-1.png)<!-- -->

``` r
risk_split <- initial_split(risk,
                            prop = 0.75,
                            strata = Risk)

risk_training <- risk_split %>% 
  training()

risk_test <- risk_split %>% 
  testing()
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

Note that the `echo = FALSE` parameter was added to the code chunk to
prevent printing of the R code that generated the plot.
