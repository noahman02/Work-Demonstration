#### SVM RBF Model ####

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(stacks)
library(tictoc)
library(doParallel)

# Handle common conflicts
tidymodels_prefer()

# load required objects ----
load("model_info/house_recipe.rda")
load("Objects/house_folds.rda")


#### Building model for svmrbf ####
SVMrbf_model <- svm_rbf(
  mode = "regression",
  cost = tune(),
  rbf_sigma = tune()
) %>%
  set_engine("kernlab", importance = impurity)
SVMrbf_wflow <- workflow() %>% add_model(SVMrbf_model) %>% add_recipe(house_recipe)
SVMrbf_params <- extract_parameter_set_dials(SVMrbf_model)
SVMrbf_grid <- grid_regular(SVMrbf_params, levels = 5)



#### Tuning process ####
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

tic("SVM RBF")
SVMrbf_tuned <- SVMrbf_wflow %>%
  tune_grid(house_folds, grid = SVMrbf_grid, control = ctrl_grid)
toc(log = TRUE)
SVMrbf_time <- tic.log(format = TRUE)

stopCluster(cl)


#### Exploring the best model ####

show_best(SVMrbf_tuned, metric = "rmse")

save(SVMrbf_tuned, file = "Objects/SVMrbf_tuned.rda")

#### Creating results for the testing data ####
load("Objects/SVMrbf_tuned.rda")


SVMrbf_wflow_tuned <- SVMrbf_wflow %>%
  finalize_workflow(select_best(SVMrbf_tuned, metric = "rmse"))

house_train <- house_split %>% training()
house_test <- house_split %>% testing()

SVMrbf_results <- fit(SVMrbf_wflow_tuned, house_train)


save(SVMrbf_results, file = "Objects/SVMrbf_results.rda")

