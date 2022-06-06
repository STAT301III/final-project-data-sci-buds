## model specs
# model workflow

# load packages----
library(tidyverse)
library(tidymodels)
library(tictoc)

# handle common conflicts
tidymodels_prefer()

# set seed
set.seed(3013)

# load recipe
load(file = "model_info/model_setup.rda")

# define models----
enet_spec <- 
  linear_reg(mode = "regression",
             penalty = tune(), 
             mixture = tune()) %>% 
  set_engine("glmnet")

knn_spec <- nearest_neighbor(mode = "regression",
                             neighbors = tune()) %>%
  set_engine("kknn") 

rf_spec <- rand_forest(mode = "regression",
                       min_n = tune(),
                       mtry = tune()) %>% 
  set_engine("ranger")

boost_spec <- boost_tree(mode = "regression",
                         mtry = tune(),
                         min_n = tune(),
                         learn_rate = tune()) %>%
  set_engine("xgboost")

svm_p_spec <- svm_poly(mode = "regression",
                       cost = tune(), 
                       degree = tune(),
                       scale_factor = tune()) %>% 
  set_engine("kernlab") 

svm_r_spec <- svm_rbf(mode = "regression",
                      cost = tune(), 
                      rbf_sigma = tune()) %>% 
  set_engine("kernlab")

nnet_spec <- mlp(mode = "regression",
                 hidden_units = tune(), 
                 penalty = tune()) %>% 
  set_engine("nnet")

mars_spec <- mars(mode = "regression",
                  num_terms = tune(),
                  prod_degree = tune()) %>%  
  set_engine("earth") 

# set up tuning grid----
enet_params <- parameters(enet_spec)
knn_params <- parameters(knn_spec)

rf_params <- parameters(rf_spec) %>% 
  update(mtry = mtry(c(1, 20))) 

boost_params <- parameters(boost_spec) %>%
  update(mtry = mtry(c(1, 20)),
         learn_rate = learn_rate(c(-5, -0.2)))

svm_p_params <- parameters(svm_p_spec)
svm_r_params <- parameters(svm_r_spec)
nnet_params <- parameters(nnet_spec)
mars_params <- parameters(mars_spec)

# define tuning grid----
enet_grid <- grid_regular(enet_params, levels = 3)
knn_grid <- grid_regular(knn_params, levels = 3)
rf_grid <- grid_regular(rf_params, levels = 3)
boost_grid <- grid_regular(boost_params, levels = 3)
svm_p_grid <- grid_regular(svm_p_params, levels = 3)
svm_r_grid <- grid_regular(svm_r_params, levels = 3)
nnet_grid <- grid_regular(nnet_params, levels = 3)
mars_grid <- grid_regular(mars_params, levels = 3)


# model workflow
enet_workflow <- workflow() %>% 
  add_model(enet_spec) %>% 
  add_recipe(superstore_rec)

knn_workflow <- workflow() %>% 
  add_model(knn_spec) %>% 
  add_recipe(superstore_rec)

rf_workflow <- workflow() %>% 
  add_model(rf_spec) %>% 
  add_recipe(superstore_rec)

boost_workflow <- workflow() %>% 
  add_model(boost_spec) %>% 
  add_recipe(superstore_rec)

svm_r_workflow <- workflow() %>% 
  add_model(svm_r_spec) %>% 
  add_recipe(superstore_rec)

svm_p_workflow <- workflow() %>% 
  add_model(svm_p_spec) %>% 
  add_recipe(superstore_rec)

nnet_workflow <- workflow() %>% 
  add_model(nnet_spec) %>% 
  add_recipe(superstore_rec)

mars_workflow <- workflow() %>% 
  add_model(mars_spec) %>% 
  add_recipe(superstore_rec)

# tuning and fitting
tic("ENET")
enet_tuned <- enet_workflow %>% 
  tune_grid(superstore_resamples, grid = enet_grid)
toc(log = TRUE)

tic("KNN")
knn_tuned <- knn_workflow %>% 
  tune_grid(superstore_resamples, grid = knn_grid)
toc(log = TRUE)

tic("RF")
rf_tuned <- rf_workflow %>% 
  tune_grid(superstore_resamples, grid = rf_grid)
toc(log = TRUE)

tic("BOOST")
boost_tuned <- boost_workflow %>% 
  tune_grid(superstore_resamples, grid = boost_grid)
toc(log = TRUE)

tic("SVMP")
svm_p_tuned <- svm_p_workflow %>% 
  tune_grid(superstore_resamples, grid = svm_p_grid)
toc(log = TRUE)

tic("SVMR")
svm_r_tuned <- svm_r_workflow %>% 
  tune_grid(superstore_resamples, grid = svm_r_grid)
toc(log = TRUE)

tic("NNET")
nnet_tuned <- nnet_workflow %>% 
  tune_grid(superstore_resamples, grid = nnet_grid)
toc(log = TRUE)

tic("MARS")
mars_tuned <- mars_workflow %>% 
  tune_grid(superstore_resamples, grid = mars_grid)
toc(log = TRUE)

# save runtime info
runtime <- tic.log(format = TRUE)
save(enet_workflow, knn_workflow, rf_workflow, boost_workflow, svm_p_workflow,
     svm_r_workflow, nnet_workflow, mars_workflow,
     enet_tuned, knn_tuned, rf_tuned, boost_tuned, svm_p_tuned, svm_r_tuned,
     nnet_tuned, mars_tuned, runtime, file = "results/tuned_models.rda")

# fitting models 

##load file
load(file = "results/tuned_models.rda")

##compare tuned models
show_best(enet_tuned, metric = "rmse") %>% head(1)
show_best(knn_tuned, metric = "rmse") %>% head(1)
show_best(rf_tuned, metric = "rmse") %>% head(1)
show_best(boost_tuned, metric = "rmse") %>% head(1)
show_best(svm_p_tuned, metric = "rmse") %>% head(1)
show_best(svm_r_tuned, metric = "rmse") %>% head(1)
show_best(nnet_tuned, metric = "rmse") %>% head(1)
show_best(mars_tuned, metric = "rmse") %>% head(1)

# svm p the lowest rmse but took the longest, rf the next best

#fit best model to training ----
svm_r_wflow_tuned <- svm_r_workflow %>% 
  finalize_workflow(select_best(svm_p_tuned, metric = "rmse"))

svm_r_results <- fit(svm_r_wflow_tuned, superstore_train)




