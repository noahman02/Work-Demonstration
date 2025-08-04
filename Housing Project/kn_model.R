#### KN Model ####

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


#### Building KN model ####
KN_model <- nearest_neighbor(mode = "regression",
                             neighbors = tune()) %>%
  set_engine("kknn")
KN_wflow <- workflow() %>% add_model(KN_model) %>% add_recipe(house_recipe)
KN_params <- extract_parameter_set_dials(KN_model)
KN_grid <- grid_regular(KN_params,levels = 5)



#### Tuninig Process ####
cl <- makePSOCKcluster(4)
registerDoParallel(cl)
tic("KN")
KN_tuned <- KN_wflow %>%
  tune_grid(house_folds, grid = KN_grid)
toc(log = TRUE)
KN_time <- tic.log(format = TRUE)

stopCluster(cl)


#### Explore model ####
show_best(KN_tuned,metric = "rmse")

save(KN_tuned, file = "Objects/KN_tuned.rda")
