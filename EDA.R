# Loading packages ----
library(tidyverse)
library(tidymodels)
library(ggplot2)

# Handle common conflicts ----
tidymodels_prefer()

# Set seed ----
set.seed(1989)

# Data import ----
load("tita_setup.Rdata")

# Create EDA data set to analyze ----
tita_eda <- tita_train %>% 
  slice_sample(prop = 0.3)

# Missingness ----
naniar::miss_var_summary(tita_eda)
  # cabin (78.5%) and age (20.4%) missing


# Outcome variable ####
skimr::skim_without_charts(tita_eda)
tita_eda %>% count(survived)
  # pretty even 


# Quantitative variables ####
ggplot(tita_eda, aes(pclass)) +
  geom_histogram(bins = 3)
  # not too skewed
  # more of factor 
ggplot(tita_eda, aes(age)) +
  geom_histogram(bins = 20)
  # pretty normal distribution, some missingness, some outliers 
ggplot(tita_eda, aes(sib_sp)) +
  geom_bar()
tita_eda %>% count(sib_sp)
  # very skewed to 0 and 1 (8 highest)
  # also more of factor 
ggplot(tita_eda, aes(parch)) +
  geom_bar()
tita_eda %>% count(parch)
  # very skewed to 0, 1, and 2 (6 highest)
  # also more of factor
tita_eda %>%
  mutate(ticket = as.numeric(ticket)) %>% 
  ggplot(aes(ticket)) + 
  geom_histogram(bins = 60)
# non finite values 
ggplot(tita_eda, aes(fare)) +
  geom_histogram(bins = 20)
  # slight right skew

# Qualitative variables ####
tita_eda %>% count(sex)
  # more males 
tita_eda %>% count(cabin) %>% view()
  # 45 and lots of missing
tita_eda %>% count(embarked)
  # C, Q, S (most S)


# NOTES ####
# make passenger_id as "id"
# NLP for name? or something with prefix?
# what to do about ticket number...
# break cabin into groups by letter? (A-G, NA)
# what to do about missingness...