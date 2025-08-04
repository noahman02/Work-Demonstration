#### Random Forest Model ####

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

## RF model build ###
RF_model <- rand_forest(
  mode = "regression",
  min_n = tune(),
  mtry = tune()
) %>%
  set_engine("ranger", importance = "impurity")
RF_wflow <- workflow() %>% add_model(RF_model) %>% add_recipe(house_recipe)
RF_params <- extract_parameter_set_dials(RF_model) %>%
  update(mtry = mtry(c(1,21)))
RF_grid <- grid_regular(RF_params, levels = 5)


## Tuning and validation step ##
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

tic("Random Forest")
RF_tuned <- RF_wflow %>%
  tune_grid(house_folds, grid = RF_grid)
toc(log = TRUE)
RF_time <- tic.log(format = TRUE)

stopCluster(cl)

## Explore best model ##
show_best(RF_tuned, metric = "rmse")

save(RF_tuned, file = "Objects/RF_tuned_refined.rda")


load("Objects/RF_tuned.rda")
RF_wflow_tuned <- RF_wflow %>%
  finalize_workflow(select_best(RF_tuned, metric = "rmse"))

house_train <- house_split %>% training()
house_test <- house_split %>% testing()

rf_results <- fit(RF_wflow_tuned, house_train)


save(rf_results, file = "Objects/rf_results.rda")
load("Objects/rf_results.rda")

rf_predicts <- house_test %>%
  select(house_price) %>%
  bind_cols(predict(rf_results, house_test))

rmse(rf_predicts,truth = house_price,estimate = .pred)


rf_results %>%
  extract_fit_parsnip() %>%
  vip(num_features = 20) + 
  labs(title = "Random forest variable importance") 
