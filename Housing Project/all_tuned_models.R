#### All Tuning Models ####


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
ctrl_grid <- control_stack_grid()



SVMrbf_model <- svm_rbf(
  mode = "regression",
  cost = tune(),
  rbf_sigma = tune()
) %>%
  set_engine("kernlab")
SVMrbf_wflow <- workflow() %>% add_model(SVMrbf_model) %>% add_recipe(house_recipe)
SVMrbf_params <- extract_parameter_set_dials(SVMrbf_model)
SVMrbf_grid <- grid_regular(SVMrbf_params, levels = 5)

SLNN_model <- mlp(
  mode = "regression",
  hidden_units = tune(),
  penalty = tune()
) %>%
  set_engine("nnet")
SLNN_wflow <- workflow() %>% add_model(SLNN_model) %>% add_recipe(house_recipe)
SLNN_params <- extract_parameter_set_dials(SLNN_model)
SLNN_grid <- grid_regular(SLNN_params, levels = 5)

RF_model <- rand_forest(
  mode = "regression",
  min_n = tune(),
  mtry = tune()
) %>%
  set_engine("ranger")
RF_wflow <- workflow() %>% add_model(RF_model) %>% add_recipe(house_recipe)
RF_params <- extract_parameter_set_dials(RF_model) %>%
  update(mtry = mtry(c(1,39)))
RF_grid <- grid_regular(RF_params, levels = 5)

MARS_model <- mars(
  mode = "regression",
  num_terms = tune(),
  prod_degree = tune()
) %>%
  set_engine("earth")
MARS_wflow <- workflow() %>% add_model(MARS_model) %>% add_recipe(house_recipe)
MARS_params <- extract_parameter_set_dials(MARS_model)
MARS_grid <- grid_regular(MARS_params, levels = 5)


KN_model <- nearest_neighbor(mode = "regression",
                             neighbors = tune()) %>%
  set_engine("kknn")
KN_wflow <- workflow() %>% add_model(KN_model) %>% add_recipe(house_recipe)
KN_params <- extract_parameter_set_dials(KN_model)
KN_grid <- grid_regular(KN_params,levels = 5)


EN_model <- linear_reg(mixture = tune(), penalty = tune()) %>%
  set_engine("glmnet")

EN_wflow <- workflow() %>% 
  add_model(EN_model) %>%
  add_recipe(house_recipe)

EN_params <- extract_parameter_set_dials(EN_model)

EN_grid <- grid_regular(EN_params, levels = 5)


BT_model <- boost_tree(
  mode = "regression",
  mtry = tune(),
  min_n = tune(),
  learn_rate = tune()
) %>%
  set_engine("xgboost")
BT_wflow <- workflow() %>% add_model(BT_model) %>% add_recipe(house_recipe)
BT_params <- extract_parameter_set_dials(BT_model) %>%
  update(mtry = mtry(c(1,39)))
BT_grid <- grid_regular(BT_params, levels = 5)


cl <- makePSOCKcluster(4)
registerDoParallel(cl)

tic("SVM RBF")
SVMrbf_tuned <- SVMrbf_wflow %>%
  tune_grid(house_folds, grid = SVMrbf_grid, control = ctrl_grid)
toc(log = TRUE)
SVMrbf_time <- tic.log(format = TRUE)

tic("Single Layer Neural Network")
SLNN_tuned <- SLNN_wflow %>%
  tune_grid(house_folds, grid = SLNN_grid,control = ctrl_grid)
toc(log = TRUE)
SLNN_time <- tic.log(format = TRUE)


tic("Random Forest")
RF_tuned <- RF_wflow %>%
  tune_grid(house_folds, grid = RF_grid, control = ctrl_grid)
toc(log = TRUE)
RF_time <- tic.log(format = TRUE)


tic("MARS")
MARS_tuned <- MARS_wflow %>%
  tune_grid(house_folds, grid = MARS_grid,control = ctrl_grid)
toc(log = TRUE)
MARS_time <- tic.log(format = TRUE)

tic("KN")
KN_tuned <- KN_wflow %>%
  tune_grid(house_folds, grid = KN_grid,control = ctrl_grid)
toc(log = TRUE)
KN_time <- tic.log(format = TRUE)

tic("Elastic Net")
EN_tuned <- EN_wflow %>%
  tune_grid(house_folds, grid = EN_grid,control = ctrl_grid)
toc(log = TRUE)
EN_time <- tic.log(format = TRUE)


tic("Boosted Tree")
BT_tuned <- BT_wflow %>%
  tune_grid(house_folds, grid = BT_grid,control = ctrl_grid)
toc(log = TRUE)
BT_time <- tic.log(format = TRUE)

stopCluster(cl)


show_best(SVMrbf_tuned, metric = "rmse")
show_best(SLNN_tuned, metric = "rmse")
show_best(RF_tuned, metric = "rmse")
show_best(MARS_tuned, metric = "rmse")
show_best(KN_tuned,metric = "rmse")
show_best(EN_tuned, metric = "rmse")
show_best(BT_tuned, metric = "rmse")

save(BT_tuned, file = "Objects/BT_tuned.rda")
save(EN_tuned, file = "Objects/EN_tuned.rda")
save(KN_tuned, file = "Objects/KN_tuned.rda")
save(MARS_tuned, file = "Objects/MARS_tuned.rda")
save(RF_tuned, file = "Objects/RF_tuned.rda")
save(SLNN_tuned, file = "Objects/SLNN_tuned.rda")
save(SVMrbf_tuned, file = "Objects/SVMrbf_tuned.rda")




house_test <- house_split %>% testing()


cl <- makePSOCKcluster(4)
registerDoParallel(cl)
parallel::clusterEvalQ(cl, {library(tidymodels)})
# Create data stack ----
house_st <- 
  stacks() %>%
  add_candidates(RF_tuned) %>%
  add_candidates(BT_tuned) %>%
  add_candidates(SVMrbf_tuned)


# Fit the stack ----
# penalty values for blending (set penalty argument when blending)
blend_penalty <- c(10^(-6:-1), 0.5, 1, 1.5, 2)

# Blend predictions using penalty defined above (tuning step, set seed)
set.seed(9876)
house_model_st <-
  house_st %>%
  blend_predictions(penalty = blend_penalty)


# Save blended model stack for reproducibility & easy reference (Rmd report)
save(house_model_st, file = "model_info/house_model_st.rda")

house_model_st <-
  house_model_st %>%
  fit_members()
theme_set(theme_bw())
autoplot(house_model_st)
autoplot(house_model_st, type = "members")
autoplot(house_model_st, type = "weights")

# fit to ensemble to entire training set ----
house_test <- 
  house_test %>%
  bind_cols(predict(house_model_st, .))
save(house_test, file = "model_info/house_test.rda")

load("model_info/house_test.rda")
ggplot(house_test) +
  aes(x = house_price, 
      y = .pred) +
  geom_point() + 
  coord_obs_pred()


member_preds <- 
  house_test %>%
  select(house_price) %>%
  bind_cols(predict(house_model_st, house_test, members = TRUE))


map_dfr(member_preds, rmse, truth = house_price, data = member_preds) %>%
  mutate(member = colnames(member_preds))



stopCluster(cl)

