## data setup 

# load package(s)
library(tidymodels)
library(tidyverse)
library(tictoc)
library(janitor)

# handle common conflicts
tidymodels_prefer()

# set seed
set.seed(3013)

# load data
superstore_dat <- read_csv(file = "data/unprocessed/Superstore_data.csv") %>% 
  clean_names() %>% 
  mutate(
    ship_mode = factor(ship_mode),
    segment = factor(segment),
    city = factor(city),
    state = factor(state),
    region = factor(region),
    category = factor(category),
    sub_category = factor(sub_category)
  )

# split data
superstore_split <- 
  initial_split(superstore_dat,
                prop = .8,
                strata = profit)

superstore_train <- training(superstore_split)
superstore_test <- testing(superstore_split)

# set folds
superstore_resamples <- 
  superstore_train %>% 
  vfold_cv(v = 5, repeats = 3, strata = profit)

# set re-sampling options 
keep_pred <- control_resamples(save_pred = T, save_workflow = T)

# set up recipe
superstore_rec <- recipe(profit ~ ., data = superstore_dat) %>%
  step_rm(row_id, order_id, order_date, ship_date, customer_id, customer_name,
          country, postal_code, product_id, product_name) %>%
  step_other(all_nominal_predictors()) %>%  
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>% 
  step_nzv(all_predictors())

superstore_rec %>%
  prep() %>%
  bake(new_data = NULL)

# save data
save(superstore_train, superstore_test, superstore_resamples, keep_pred, 
     superstore_rec, file = "model_info/dat_setup.rda")
