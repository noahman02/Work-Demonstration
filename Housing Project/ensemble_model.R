#### Ensemble Model ####

# Load package(s) ----
library(tidymodels)
library(tidyverse)
library(stacks)

# Handle common conflicts
tidymodels_prefer()

# Load candidate model info ----
load("Objects/BT_tuned.rda")
load("Objects/RF_tuned.rda")
load("Objects/KN_tuned.rda")
load("Objects/MARS_tuned.rda")
load("Objects/SLNN_tuned.rda")
load("Objects/EN_tuned.rda")
load("Objects/SVMrbf_tuned.rda")



show_best(RF_tuned)
show_best(BT_tuned)
show_best(KN_tuned)
show_best(EN_tuned)
show_best(MARS_tuned)
show_best(SLNN_tuned)
show_best(SVMrbf_tuned)

# BT RF, and SVM_rbf models performed the best, so I will make an ensemble model of both


# Load split data object & get testing data
load("Objects/house_split.rda")

house_test <- house_split %>% testing()

# Create data stack ----
house_st <- 
  stacks() %>%
  add_candidates(RF_tuned) %>%
  add_candidates(BT_tuned) %>%
  add_candidates(SVMrbf_tuned)

save(house_st,file = "Objects/house_st_RF_BT.rda")

# Fit the stack ----

# penalty values for blending (set penalty argument when blending)
blend_penalty <- c(10^(-6:-1), 0.5, 1, 1.5, 2)

# Blend predictions using penalty defined above (tuning step, set seed)
set.seed(9876)
house_model_st <-
  house_st %>%
  blend_predictions(penalty = blend_penalty)


# Save blended model stack for reproducibility & easy reference (Rmd report)
save(house_model_st, file = "model_info/house_model_st2.rda")

# Explore the blended model stack
load("model_info/house_model_st.rda")
house_model_st <-
  house_model_st %>%
  fit_members()
save(house_model_st,file ="model_info/house_model_st_fitted.rda")
theme_set(theme_bw())
autoplot(house_model_st)
autoplot(house_model_st, type = "members")
autoplot(house_model_st, type = "weights")

# fit to ensemble to entire training set ----
house_test <- 
  house_test %>%
  bind_cols(predict(house_model_st, .))


# Save trained ensemble model for reproducibility & easy reference (Rmd report)
save(house_test, file = "model_info/house_test2.rda")

# Explore and assess trained ensemble model
load("model_info/house_test2.rda")
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

