if (!require("pacman"))
  install.packages("pacman")
pacman::p_load(
  tidyverse,
  tidymodels,
  caret,
  lubridate,
  zeallot,
  xgboost,
  cowplot,
  doParallel,
  Cubist,
  pls,
  earth,
  elasticnet,
  ipred,
  plyr,
  e1071,
  kernlab,
  randomForest,
  ParBayesianOptimization,
  xgboost
)


# Data --------------------------------------------------------------------

# The assessment results with respect to MR1
load("./mt_results/mr1_mt.Rda")


# Function ----------------------------------------------------------------

# functions to compute the proportion of the assessment results;
# the assessment results are labelled using numbers 1 to 4
get_consistency_rate <- function(x) {
  (sum(x == 4)) / length(x)
}

get_inconsistency_rate <- function(x) {
  (sum(x == 3)) / length(x)
}

get_invalid_rate <- function(x) {
  (sum(x == 2)) / length(x)
}

get_inconclusive_rate <- function(x) {
  (sum(x == 1)) / length(x)
}

get_evaluation_df <- function(xs, change_ratio = (50:150) * 0.01) {
  # xs: assessment results of model obtained during an outer CV iteration, which is a list that each store the results
  # associated with a change ratio 
  # change_ratio: the ratio of the precipitation compared to that of the original samples
  tibble(
    change_ratio = change_ratio,
    Consistent = sapply(xs, get_consistency_rate),
    Inconsistent = sapply(xs, get_inconsistency_rate),
    Invalid = sapply(xs, get_invalid_rate),
    Inconclusive = sapply(xs, get_inconclusive_rate)
  ) %>%
    gather(outcome, value, -change_ratio)
}

# "eval_grid" store all the assessment results; loaded from "mr1_mt.Rda"
# "evaluation_tes" and "evaluation_trs" columns are added to store the results of test and training set
eval_grid <- eval_grid %>%
  mutate(evaluation_tes = vector("list", 1),
         evaluation_trs = vector("list", 1))

# iterate over region, season, data splits
for (i in 1:nrow(eval_grid)) {
  # extract the assessment results for training and test sets, and store them in "mt_trs" and "mt_tes"
  mt_trs <- eval_grid$mt_trs[[i]]
  mt_tes <- eval_grid$mt_tes[[i]]
  
  # "evaluation_tr" and "evaluation_te" store the results for each model
  evaluation_tr <- tibble(model = names(mt_trs),
                          evaluation = vector("list", 1))
  evaluation_te <- tibble(model = names(mt_tes),
                          evaluation = vector("list", 1))
  
  # iteration over machine learning models
  for (j in 1:length(mt_trs)) {
    # extract results and compute the portions of assessment results
    mt_tr <- mt_trs[[j]]
    mt_te <- mt_tes[[j]]
    
    evaluation_tr$evaluation[[j]] <- get_evaluation_df(mt_tr)
    evaluation_te$evaluation[[j]] <- get_evaluation_df(mt_te)
  }
  
  # store the result of all models in "eval_grid"
  eval_grid$evaluation_trs[[i]] <- evaluation_tr %>%
    unnest(evaluation)
  eval_grid$evaluation_tes[[i]] <- evaluation_te %>%
    unnest(evaluation)
}


# Plot --------------------------------------------------------------------


# prepare plot tibble

# "data_gof" save the goodness of fit results for ranking the models;
# the results are stored in "model_order"
data_gof <- eval_grid %>%
  select(region, season, iter, gof_result) %>%
  unnest(gof_result)

model_order <- data_gof %>%
  group_by(model) %>%
  dplyr::summarise(mean_gof = mean(r2)) %>%
  arrange(desc(mean_gof)) %>%
  pull(model)

# extract results from "eval_grid" and label and ordering the results
data_plot <- eval_grid %>%
  select(region, season, iter, evaluation_tes) %>% # plot the result for the test set only
  unnest(cols = evaluation_tes) %>%
  mutate(outcome = factor(
    outcome,
    levels = c("Consistent", "Inconsistent", "Inconclusive", "Invalid"),
    labels = c("Consistent", "Inconsistent", "Inconclusive test", "Invalid")
  )) %>%
  mutate(model = factor(model, levels = model_order))


# Plotting results on summer floods ---------------------------------------

# select results associated with summer floods
data_plot2 <- data_plot %>%
  dplyr::filter(region %in% c(1:4),
                season %in% c("S")) %>% # filtering summer floods
  group_by(region, season, model, change_ratio, outcome) %>%
  dplyr::summarise(value = mean(value)) %>%
  mutate(region = paste0("Region ", region))

ggplot(data_plot2, aes(change_ratio*100, value, fill = outcome)) +
  geom_area(alpha = 0.9) +
  scale_fill_manual(values = c(
    "#a3d977",
    "#ffc374",
    "#d0d0d0",
    "#ffbcb2"
    )) + # the color matches the assessment results
  scale_x_continuous(breaks = c(50, 100, 150), expand = c(0, 0)) +
  scale_y_continuous(breaks = c(0, 0.2, 0.4, 0.6, 0.8, 1),
                     expand = c(0, 0)) +
  facet_grid(region~model)+
  labs(fill = "Assessment result",
       x = "Magnitude of precipitation compared to the precipitation of the orignal input sample [%]",
       y = "Proportion of assessment result") +
  guides(fill = guide_legend(reverse=TRUE)) +
  theme_bw(base_size = 8)+
  theme(
    legend.position = "top",
    strip.background = element_rect(fill = "grey80", size = 0),
    strip.text.y = element_text(),
    panel.spacing.x = unit(0.4, "lines"),
    panel.spacing.y = unit(0.5, "lines"),
    panel.border = element_blank(),
    panel.grid = element_line(color = "grey20"),
    axis.text.x = element_text(angle = 90)
  )

ggsave(
  filename = "./paper_figures/Figure8.png",
  width = 7,
  height = 5,
  units = "in",
  dpi = 600
)


# Plotting results on winter floods ---------------------------------------


data_plot2 <- data_plot %>%
  dplyr::filter(region %in% c(1:4),
                season %in% c("W")) %>% # filtering winter floods
  group_by(region, season, model, change_ratio, outcome) %>%
  dplyr::summarise(value = mean(value)) %>%
  mutate(region = paste0("Region ", region))

ggplot(data_plot2, aes(change_ratio*100, value, fill = outcome)) +
  geom_area(alpha = 0.9) +
  scale_fill_manual(values = c(
    "#a3d977",
    "#ffc374",
    "#d0d0d0",
    "#ffbcb2"
  )) +
  scale_x_continuous(breaks = c(50, 100, 150), expand = c(0, 0))+
  scale_y_continuous(breaks = c(0, 0.2, 0.4, 0.6, 0.8, 1),
                     expand = c(0, 0)) +
  facet_grid(region~model) +
  labs(fill = "Assessment result",
       x = "Magnitude of precipitation compared to the precipitation of the orignal input sample [%]",
       y = "Proportion of assessment result") +
  guides(fill = guide_legend(reverse=TRUE)) +
  theme_bw(base_size = 8)+
  theme(
    legend.position = "top",
    strip.background = element_rect(fill = "grey80", size = 0),
    strip.text.y = element_text(),
    panel.spacing.x = unit(0.4, "lines"),
    panel.spacing.y = unit(0.5, "lines"),
    panel.border = element_blank(),
    panel.grid = element_line(color = "grey20"),
    axis.text.x = element_text(angle = 90)
  )

ggsave(
  filename = "./paper_figures/Figure_S7.png",
  width = 7,
  height = 5,
  units = "in",
  dpi = 600
)

