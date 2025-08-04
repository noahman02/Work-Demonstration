#### Boosted Tree Model refined ####
# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(stacks)
library(tictoc)
library(doParallel)

# Handle common conflicts
tidymodels_prefer()
ctrl_grid <- control_stack_grid()


#### New Recipe and Model ####
# load required objects ----
load("model_info/house_recipe_refined.rda")
load("Objects/house_folds2.rda")



#### New Boosted Tree Model ####
BT_model <- boost_tree(
  mode = "regression",
  mtry = tune(),
  min_n = tune(),
  learn_rate = tune()
) %>%
  set_engine("xgboost", importance = "impurity")
BT_wflow <- workflow() %>% add_model(BT_model) %>% add_recipe(house_recipe_refined)
BT_params <- extract_parameter_set_dials(BT_model) %>%
  update(mtry = mtry(c(1,23)))
BT_grid <- grid_regular(BT_params, levels = 5)



#### Validation and tuning parameters ####
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

tic("Boosted Tree")
BT_tuned <- BT_wflow %>%
  tune_grid(house_folds, grid = BT_grid, control = ctrl_grid)
toc(log = TRUE)
BT_time <- tic.log(format = TRUE)


stopCluster(cl)

#### Selecting and Exploring best boosted tree model ####
show_best(BT_tuned, metric = "rmse")

save(BT_tuned, file = "Objects/BT_tuned_refined.rda")

load("Objects/BT_tuned_refined.rda")

BT_wflow_tuned <- BT_wflow %>%
  finalize_workflow(select_best(BT_tuned, metric = "rmse"))

house_train <- house_split %>% training()
house_test <- house_split %>% testing()

BT_results <- fit(BT_wflow_tuned, house_train)


save(BT_results, file = "Objects/BT_results_refined.rda")