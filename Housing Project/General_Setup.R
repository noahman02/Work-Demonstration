#### Selecting Variables ####


#### Load Libraries ####
library(tidyverse)
library(tidymodels)
library(skimr)
library(patchwork)
library(kableExtra)
library(broom)


# Handle common conflicts
tidymodels_prefer()

# Seed
set.seed(3013)


#### Load Data ####
Housing <-
  distinct(
    read_csv("data/processed/housing.csv") %>% filter(!is.na(Bedrooms)) %>% select(
      -c(
        `Tax Year`,
        `Sale Date`,
        `Deed No.`,
        `Sale Price`,
        `Attic Finish`,
        Renovation,
        `Census Tract`
      )
    ) %>%
      filter(
        `Town Code` == 17 &
          `Prior Tax Year Market Value Estimate (Building)` != 0
      ) %>%
      mutate(
        Porch = ifelse(is.na(Porch),
                       0,
                       Porch),
        House_Price = `Prior Tax Year Market Value Estimate (Land)` + `Prior Tax Year Market Value Estimate (Building)`,
        Non_Bedrooms = Rooms - Bedrooms
      ) %>% select(
        -c(
          `Prior Tax Year Market Value Estimate (Building)`,
          `Prior Tax Year Market Value Estimate (Land)`,
          `Multi Property Indicator`,
          `Property Address`,
          Use,
          `Percent Ownership`,
          `Condo Class Factor`,
          `Repair Condition`,
          `Multi Code`,
          `Other Improvements`,
          `Design Plan`,
          Rooms,
          `New Georeferenced Column`,
          `Modeling Group`,
          PIN,
          `Total Building Square Feet`,
          `Multi-Family Indicator`
        )
      ) %>% mutate(Non_Bedrooms = ifelse(Non_Bedrooms < 0, 0, Non_Bedrooms))
  ) %>% janitor::clean_names()

save(Housing, file ="data/processed/Housing2.csv")

#### linear model ####
# Indicates which variables are significant from a linear model
variables <- as.list(tidy(lm(house_price ~ ., Housing)) %>% filter(p.value <.05) %>% filter(term != "(Intercept)") %>% select(term))

#### Selecting Final Variables ####
house <- Housing %>% select(variables$term,house_price) %>%
  mutate(property_class = as.factor(property_class),
         neighborhood_code = as.factor(neighborhood_code),
         roof_material = as.factor(roof_material),
         basement = as.factor(basement),
         central_heating = as.factor(central_heating),
         central_air = as.factor(central_air),
         fireplaces = as.factor(fireplaces),
         site_desireability = as.ordered(site_desireability),
         garage_1_size = as.factor(garage_1_size),
         garage_1_material = as.factor(garage_1_material),
         garage_1_attachment = as.factor(garage_1_attachment),
         porch = as.factor(porch),
         number_of_commercial_units = as.factor(number_of_commercial_units),
         large_lot = as.factor(large_lot)
         )
save(house,file = "data/processed/house.csv")

# Initial split & folding ----
house_split <- house %>%
  initial_split(prop = 0.8,strata = property_class)

house_train <- training(house_split)
house_test <- testing(house_split)

# Folds
house_folds <- vfold_cv(house_train, v = 5, repeats = 3, strata = property_class)

## Build general recipe (featuring eng.) ----
house_recipe <- recipe(house_price ~ . , data = house_train) %>%
  step_dummy(all_nominal_predictors())%>%
  step_nzv(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_interact(house_price ~ land_square_feet:building_square_feet)


## This equation will be used as a new attempt to lower the RMSE value of our best models
house_recipe_refined <-
  recipe(
    house_price ~ land_square_feet + building_square_feet + longitude + latitude + full_baths + age + property_class + bedrooms + neighborhood_code,
    house_train
  )  %>%
  step_dummy(all_nominal_predictors()) %>%
  step_nzv(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_interact(house_price ~ land_square_feet:building_square_feet)

# # Check how recipe works
house_recipe %>%
  prep(house_train) %>%
  bake(new_data = NULL)

# # Check how recipe refined works
house_recipe_refined %>%
  prep(house_train) %>%
  bake(new_data = NULL)

# objects required for tuning
# data objects
save(house_folds, file = "Objects/house_folds2.rda")
save(house_split, file = "Objects/house_split2.rda")

# model info object
save(house_recipe, house_split, file = "model_info/house_recipe.rda")
save(house_recipe_refined,house_split, file = "model_info/house_recipe_refined.rda")
