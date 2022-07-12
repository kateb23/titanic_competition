# penalty mixture .metric .estimator  mean     n std_err .config              
# <dbl>   <dbl> <chr>   <chr>      <dbl> <int>   <dbl> <chr>                
#   1    1       0    roc_auc binary     0.817    15 0.00937 Preprocessor1_Model01
# 2    1.78    0    roc_auc binary     0.814    15 0.00944 Preprocessor1_Model02
# 3    3.16    0    roc_auc binary     0.813    15 0.00948 Preprocessor1_Model03

# penalty mixture .metric  .estimator  mean     n  std_err .config              
# <dbl>   <dbl> <chr>    <chr>      <dbl> <int>    <dbl> <chr>                
#   1    1       0    accuracy binary     0.702    15 0.00621  Preprocessor1_Model01
# 2    1.78    0    accuracy binary     0.624    15 0.00325  Preprocessor1_Model02
# 3    3.16    0    accuracy binary     0.618    15 0.000798 Preprocessor1_Model03

# Loading packages ----
library(tidyverse)
library(tidymodels)
library(doMC)
library(tictoc)

# Handle common conflicts ----
tidymodels_prefer()

# Set seed ----
set.seed(1989)

# Data import ----
load("tita_setup.Rdata")

# Register cores/threads for parallel processing ----
registerDoMC(cores = parallel::detectCores(logical = TRUE))

# Recipe tuning ----
lm_base_recipe <- 
  recipe(survived ~ 
           passenger_id +
           pclass +
           # name +
           sex +
           age +
           sib_sp +
           parch +
           # ticket +
           fare +
           # cabin +
           embarked, data = tita_train) %>%
  update_role(passenger_id, new_role = "id") %>%
  step_impute_mean(age) %>%
  step_mutate(survived = as_factor(survived),
              pclass = ordered(pclass),
              sex = as_factor(sex),
              sib_sp = as_factor(sib_sp),
              parch = as_factor(parch),
              embarked = as_factor(embarked)) %>%
              # cabin = fct_collapse(cabin,
              #                      "A" = contains("A"),
              #                      "B" = contains("B"),
              #                      "C" = contains("C"),
              #                      "D" = contains("D"),
              #                      "E" = contains("E"),
              #                      "F" = contains("F"),
              #                      "oth_cab" = contains("G"))) %>%
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_nzv(all_numeric_predictors())
# potentially add interactions

lm_base_recipe %>% 
  prep() %>% 
  bake(new_data = NULL) %>% 
  slice_sample(n = 10)

# Model specifications ----
lm_base_model <-
  logistic_reg(mode = "classification",
               mixture = tune(),
               penalty = tune()) %>%
  set_engine("glmnet")

# Create Regular Grids ----
lm_base_params <- extract_parameter_set_dials(lm_base_model) %>%
  update(mixture = mixture(c(0, 1)))
lm_base_grid <- grid_regular(lm_base_params, levels = 5)

# Create workflow ----
lm_base_workflow <- workflow() %>%
  add_model(lm_base_model) %>%
  add_recipe(lm_base_recipe)

# Set metrics ---- 
tita_metrics <- metric_set(accuracy, roc_auc)

# Tune parameters to optimize model performance ----
tic.clearlog()
tic("Logistic Regression Base")
lm_base_tuned <- lm_base_workflow %>%
  tune_grid(tita_folds, 
            grid = lm_base_grid, 
            control = keep_pred,
            metrics = tita_metrics)
toc(log = TRUE)

time_log <- tic.log(format = FALSE)

lm_base_tictoc <- tibble(
  model_type = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time,
  num_models = dim(lm_base_grid)[1]
)

# Save results ----
save(lm_base_tuned, 
     lm_base_tictoc,
     file = "models/logistic_regression/lm_results/lm_base_tuned.Rdata")

load("models/logistic_regression/lm_results/lm_base_tuned.Rdata")
collect_metrics(lm_base_tuned) %>%
  filter(.metric == "roc_auc") %>%
  arrange(desc(mean))

collect_metrics(lm_base_tuned) %>%
  filter(.metric == "accuracy") %>%
  arrange(desc(mean))

autoplot(lm_base_tuned, metric = "roc_auc")
# significant drop as penalty increases past 0.00316
# biggest drop with larger mixture, smaller drop with smaller penalty (best 0, worst 1)
# tuning - smaller mixture and penalty 