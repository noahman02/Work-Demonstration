#### Null Model ####

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

keep_pred <-
  control_resamples(save_pred = TRUE, save_workflow = TRUE)


#### Building a null model ####
nl_model <- null_model(mode = "regression") %>% set_engine("parsnip")
nl_workflow <- workflow() %>% 
  add_model(nl_model) %>%
  add_recipe(house_recipe)

#### tuning process ####
cl <- makePSOCKcluster(4)
registerDoParallel(cl)
nl_res <- nl_workflow %>% 
  fit_resamples(resamples = house_folds, control = keep_pred)
stopCluster(cl)


#### Creating null model results for testing data ####
save(nl_res,file = "Objects/nl_res.rda")

house_train <- house_split %>% training()
house_test <- house_split %>% testing()

nl_results <- fit(nl_workflow, house_train)
nl_predicts <- house_test %>%
  select(house_price) %>%
  bind_cols(predict(nl_results, house_test))

save(nl_predicts,file = "model_info/nl_predicts.rda")

rmse(nl_predicts,truth = house_price,estimate = .pred)
