# mtry min_n learn_rate .metric  .estimator  mean     n std_err .config               
# <int> <int>      <dbl> <chr>    <chr>      <dbl> <int>   <dbl> <chr>                 
#   1     5     2     0.316  accuracy binary     0.817    15 0.00565 Preprocessor1_Model101
# 2    16     2     0.316  accuracy binary     0.815    15 0.00763 Preprocessor1_Model102
# 3    27     2     0.316  accuracy binary     0.815    15 0.00763 Preprocessor1_Model103

# mtry min_n learn_rate .metric .estimator  mean     n std_err .config               
# <int> <int>      <dbl> <chr>   <chr>      <dbl> <int>   <dbl> <chr>                 
#   1     5     2    0.001   roc_auc binary     0.849    15 0.00780 Preprocessor1_Model001
# 2     5     2    0.00422 roc_auc binary     0.847    15 0.00793 Preprocessor1_Model026
# 3     5     2    0.316   roc_auc binary     0.846    15 0.00716 Preprocessor1_Model101

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
bt_base_recipe <- 
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
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  # step_normalize(all_numeric_predictors()) %>%
  step_nzv(all_numeric_predictors())

# Boosted tree model specification ----
bt_base_model <- boost_tree(mode = "classification",
                                 min_n = tune(),
                                 mtry = tune(),
                                 learn_rate = tune()) %>%
  set_engine("xgboost")

# Create regular grids ----
bt_base_params <- extract_parameter_set_dials(bt_base_model)%>%
  update(mtry = mtry(c(5, 50)))
bt_base_grid <- grid_regular(bt_base_params, levels = 5)

# Create workflow ----
bt_base_workflow <- workflow() %>%
  add_model(bt_base_model) %>%
  add_recipe(bt_base_recipe)

# Set metrics ----
tita_metrics <- metric_set(accuracy, roc_auc)

# Tune parameters to optimize model performance ----
tic.clearlog()
tic("Boosted Tree Base")
bt_base_tuned <- bt_base_workflow %>%
  tune_grid(tita_folds, 
            grid = bt_base_grid, 
            control = keep_pred,
            metrics = tita_metrics)
toc(log = TRUE)

time_log <- tic.log(format = FALSE)

bt_base_tictoc <- tibble(
  model_type = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time,
  num_models = dim(bt_base_grid)[1]
)

# Save results ----
save(bt_base_tuned, 
     bt_base_tictoc,
     file = "models/boosted_trees/bt_results/bt_base_tuned.Rdata")

load("models/boosted_trees/bt_results/bt_base_tuned.Rdata")
collect_metrics(bt_base_tuned) %>%
  filter(.metric == "accuracy") %>%
  arrange(desc(mean))

collect_metrics(bt_base_tuned) %>%
  filter(.metric == "roc_auc") %>%
  arrange(desc(mean))

autoplot(bt_base_tuned, metric = "roc_auc")
# best min_n 2 (smaller)
# mtry best at 5, stagnates larger
# smaller learning rates better
