#### SLNN Model ####

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


#### Creating a single layer neural network model ####
SLNN_model <- mlp(
  mode = "regression",
  hidden_units = tune(),
  penalty = tune()
) %>%
  set_engine("nnet")
SLNN_wflow <- workflow() %>% add_model(SLNN_model) %>% add_recipe(house_recipe)
SLNN_params <- extract_parameter_set_dials(SLNN_model)
SLNN_grid <- grid_regular(SLNN_params, levels = 5)


#### Tuning process ####
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

tic("Single Layer Neural Network")
SLNN_tuned <- SLNN_wflow %>%
  tune_grid(house_folds, grid = SLNN_grid)
toc(log = TRUE)
SLNN_time <- tic.log(format = TRUE)

stopCluster(cl)

#### exploring best model ####
show_best(SLNN_tuned, metric = "rmse")

save(SLNN_tuned, file = "Objects/SLNN_tuned.rda")

