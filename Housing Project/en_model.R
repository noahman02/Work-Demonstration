#### GLM model ####
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


## Elastic Net model build ####
EN_model <- linear_reg(mixture = tune(), penalty = tune()) %>%
  set_engine("glmnet")

EN_wflow <- workflow() %>% 
  add_model(EN_model) %>%
  add_recipe(house_recipe)

EN_params <- extract_parameter_set_dials(EN_model)

EN_grid <- grid_regular(EN_params, levels = 5)


#### Tuning Process ####
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

tic("Elastic Net")
EN_tuned <- EN_wflow %>%
  tune_grid(house_folds, grid = EN_grid)
toc(log = TRUE)
EN_time <- tic.log(format = TRUE)

stopCluster(cl)


#### Explore best model and save ####
show_best(EN_tuned, metric = "rmse")

save(EN_tuned, file = "Objects/EN_tuned.rda")
