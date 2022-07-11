# Loading packages ----
library(tidyverse)
library(tidymodels)
library(ggplot2)

# Handle common conflicts ----
tidymodels_prefer()

# Set seed ----
set.seed(1989)

# Data import ----
load("tit_setup.Rdata")

# Create EDA data set to analyze ----
tit_eda <- tit_train %>% 
  slice_sample(prop = 0.3)

# Missingness ----
naniar::miss_var_summary(tit_eda)
  # cabin (78.5%) and age (20.4%) missing


# Outcome variable ####
skimr::skim_without_charts(tit_eda)
tit_eda %>% count(survived)
  # pretty even 


# Quantitative variables ####
ggplot(tit_eda, aes(pclass)) +
  geom_histogram(bins = 3)
  # not too skewed
  # more of factor 
ggplot(tit_eda, aes(age)) +
  geom_histogram(bins = 20)
  # pretty normal distribution, some missingness, some outliers 
ggplot(tit_eda, aes(sib_sp)) +
  geom_bar()
tit_eda %>% count(sib_sp)
  # very skewed to 0 and 1 (8 highest)
  # also more of factor 
ggplot(tit_eda, aes(parch)) +
  geom_bar()
tit_eda %>% count(parch)
  # very skewed to 0, 1, and 2 (6 highest)
  # also more of factor
tit_eda %>%
  mutate(ticket = as.numeric(ticket)) %>% 
  ggplot(aes(ticket)) + 
  geom_histogram(bins = 60)
# non finite values 

# Qualitative variables ####
tit_eda %>% count(sex)
  # more males 


# NOTES ####
# make passenger_id as "id"
# NLP for name?
# what to do about ticket number...