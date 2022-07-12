# Loading packages ----
library(tidyverse)
library(tidymodels)

# Handle common conflicts ----
tidymodels_prefer()

# Set seed ----
set.seed(1989)

# Data import ----
train <- read_csv("data/train.csv") %>%
  janitor::clean_names()
test <- read_csv("data/test.csv") %>%
  janitor::clean_names()

# Create testing set from training data ----
tita_split <- initial_split(train, prop = 0.7, strata = survived) 
tita_train <- training(tita_split)
tita_test <- testing(tita_split)

# Creating folds, setting control ----
tita_folds <- vfold_cv(tita_train, v = 5, strata = survived, repeats = 3)
keep_pred <- control_resamples(verbose = TRUE,
                               save_pred = TRUE, 
                               save_workflow = TRUE)

# Save data import ----
save(tita_train,
     tita_test,
     tita_folds,
     keep_pred,
     file = "tita_setup.Rdata")
