#eda 

#load packages 
library(tidyverse)
library(tidymodels)
library(janitor)
library(naniar)

#read in 
superstore_dat <- read_csv(file = "data/unprocessed/Superstore_data.csv") %>% 
  clean_names()

# initial split 
#split data
superstore_split <- 
  initial_split(superstore_dat,
                prop = .8,
                strata = profit)

superstore_train <- training(superstore_split)
superstore_test <- testing(superstore_split)

#folds
superstore_resamples <- 
  superstore_train %>% 
  vfold_cv(v = 5, repeats = 3, strata = profit)


# initial overview of data ----

#9994 observations and 21 features 
dim(superstore_dat)

#check for missingness 
miss_var_table <- miss_var_table(superstore_train)
miss_var_summary <- miss_var_summary(superstore_train)
gg_miss_var <- gg_miss_var(superstore_train)

#no missingness in the data 
miss_var_table
miss_var_summary
gg_miss_var

# essential findings  ----


# secondary findings ----