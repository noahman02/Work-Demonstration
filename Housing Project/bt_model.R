#### Boosted Tree Model ####

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

BT_model <- boost_tree(
  mode = "regression",
  mtry = tune(),
  min_n = tune(),
  learn_rate = tune()
) %>%
  set_engine("xgboost", importance = "impurity")
BT_wflow <- workflow() %>% add_model(BT_model) %>% add_recipe(house_recipe)
BT_params <- extract_parameter_set_dials(BT_model) %>%
  update(mtry = mtry(c(1,21)))
BT_grid <- grid_regular(BT_params, levels = 5)


cl <- makePSOCKcluster(4)
registerDoParallel(cl)

tic("Boosted Tree")
BT_tuned <- BT_wflow %>%
  tune_grid(house_folds, grid = BT_grid)
toc(log = TRUE)
BT_time <- tic.log(format = TRUE)


stopCluster(cl)


show_best(BT_tuned, metric = "rmse")

save(BT_tuned, file = "Objects/BT_tuned.rda")

load("Objects/BT_tuned.rda")

BT_wflow_tuned <- BT_wflow %>%
  finalize_workflow(select_best(BT_tuned, metric = "rmse"))

house_train <- house_split %>% training()
house_test <- house_split %>% testing()

BT_results <- fit(BT_wflow_tuned, house_train)


save(BT_results, file = "Objects/BT_results.rda")

BT_results %>%
  extract_fit_parsnip() %>%
  vip(num_features = 20) + 
  labs(title = "BT variable importance") 