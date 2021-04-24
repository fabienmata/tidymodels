Credit Risk with tidymodels
================
Fabien Mata
23/04/2021

### Load tidymodels packages

``` r
library(tidyverse)
library(rsample)
library(recipes)
library(parsnip)
library(yardstick)
library(naniar)
library(finalfit)
```

The data is from kaggle (‘german credit risk’) It contains 10 variables,
where

``` r
risk <- read.csv("https://raw.githubusercontent.com/fabienmata/tidymodels/master/data/german_credit_data.csv", row.names = 'X')
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

which variable has missing values

``` r
risk %>% gg_miss_var()
```

![](rmd_files/figure-gfm/naniar-1.png)<!-- -->

see clearly the missing data points

``` r
risk %>% missing_plot()
```

![](rmd_files/figure-gfm/finalfit-1.png)<!-- -->

![](rmd_files/figure-gfm/pressure-1.png)<!-- -->

Note that the `echo = FALSE` parameter was added to the code chunk to
prevent printing of the R code that generated the plot.
