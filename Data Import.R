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
tit_split <- initial_split(train, prop = 0.7, strata = survived) 
tit_train <- training(tit_split)
tit_test <- testing(tit_split)

# Creating folds, setting control ----
tit_folds <- vfold_cv(tit_train, v = 5, strata = survived, repeats = 3)
keep_pred <- control_resamples(verbose = TRUE,
                               save_pred = TRUE, 
                               save_workflow = TRUE)

# Save data import ----
save(tit_train,
     tit_test,
     tit_folds,
     keep_pred,
     file = "tit_setup.Rdata")
