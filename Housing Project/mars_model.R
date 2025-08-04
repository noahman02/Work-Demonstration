#### MARS Model ####

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


#### Building a MARS model #####
MARS_model <- mars(
  mode = "regression",
  num_terms = tune(),
  prod_degree = tune()
) %>%
  set_engine("earth")
MARS_wflow <- workflow() %>% add_model(MARS_model) %>% add_recipe(house_recipe)
MARS_params <- extract_parameter_set_dials(MARS_model)
MARS_grid <- grid_regular(MARS_params, levels = 5)



#### Tuning Process ####
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

tic("MARS")
MARS_tuned <- MARS_wflow %>%
  tune_grid(house_folds, grid = MARS_grid)
toc(log = TRUE)
MARS_time <- tic.log(format = TRUE)

stopCluster(cl)

#### exploring the best model ####
show_best(MARS_tuned, metric = "rmse")

save(MARS_tuned, file = "Objects/MARS_tuned.rda")

