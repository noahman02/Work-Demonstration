#### SVM_rbf Model refined ####


# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(stacks)
library(tictoc)
library(doParallel)

# Handle common conflicts
tidymodels_prefer()
ctrl_grid <- control_stack_grid()


# load required objects ----
load("model_info/house_recipe_refined.rda")
load("Objects/house_folds2.rda")


#### Creating model ####
SVMrbf_model <- svm_rbf(
  mode = "regression",
  cost = tune(),
  rbf_sigma = tune()
) %>%
  set_engine("kernlab", importance = impurity)
SVMrbf_wflow <- workflow() %>% add_model(SVMrbf_model) %>% add_recipe(house_recipe_refined)
SVMrbf_params <- extract_parameter_set_dials(SVMrbf_model)
SVMrbf_grid <- grid_regular(SVMrbf_params, levels = 5)


#### tuning process ####
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

tic("SVM RBF")
SVMrbf_tuned <- SVMrbf_wflow %>%
  tune_grid(house_folds, grid = SVMrbf_grid)
toc(log = TRUE)
SVMrbf_time <- tic.log(format = TRUE)

stopCluster(cl)

#### exploring the best model ###
show_best(SVMrbf_tuned, metric = "rmse")

save(SVMrbf_tuned, file = "Objects/SVMrbf_tuned_refined.rda")


#### creating results for testing data set ####
load("Objects/SVMrbf_tuned_refined.rda")

SVMrbf_wflow_tuned <- SVMrbf_wflow %>%
  finalize_workflow(select_best(SVMrbf_tuned, metric = "rmse"))

house_train <- house_split %>% training()
house_test <- house_split %>% testing()

SVMrbf_results <- fit(SVMrbf_wflow_tuned, house_train)


save(SVMrbf_results, file = "Objects/SVMrbf_results_refined.rda")

