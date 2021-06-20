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
  xgboost,
  twosamples
)

# Data --------------------------------------------------------------------

# Load the assessment results with respect MR1
load("./mt_results/mr1_mt.Rda")


# Function ----------------------------------------------------------------


# The consistent assessment result is labelled as 4
get_consistency_rate <- function(x) {
  (sum(x == 4)) / length(x)
}

# compute the consistency rate associated with a sample from the training or the test set
get_consistency_rate_per_event <- function(xs) {
  # xs: a list that store the assessment result of a model during a outer CV iteration
  
  m <- xs %>% 
    unlist() %>% 
    matrix(nrow = length(xs), byrow = T) # row: different variations from the original events; col: different events
  
  apply(m, 2, get_consistency_rate) # compute the mean consistency rate associated with each event
}

dist_test <- function(data_test1,
                      data_test2,
                      confidence_level = 0.05) {
  
  # wrapper to perform two-sample Cramer-von Mises Test
  test_result <- cvm_test(data_test1, data_test2)
  
  if (test_result[2] <= confidence_level) {
    "reject H0"
  } else {
    "fail to reject H0"
  }
}

dist_test_df <- function(df) {
  # extract data from training and the test sets and perform two-sample Cramer-von Mises Test
  data_test1 <- df %>%
    dplyr::filter(case == "Test set") %>%
    pull(consistency_rate_distribution)
  
  data_test2 <- df %>%
    dplyr::filter(case == "Training set") %>%
    pull(consistency_rate_distribution)
  
  dist_test(data_test1, data_test2)
}

# Plot --------------------------------------------------------------------

# add a column "consistency_rate_distribution" to "eval_grid" to store the average consistency rates
eval_grid <- eval_grid %>%
  mutate(consistency_rate_distribution = vector("list", 1))

# iterate over region, season, data splits
for (i in 1:nrow(eval_grid)) {
  # iterate over the rows of eval_grid, which is an outer CV iteration for a regional-season dataset
  
  mt_trs <- eval_grid$mt_trs[[i]]
  mt_tes <- eval_grid$mt_tes[[i]]
  
  # iteration over machine learning methods; "consistency_rate_distribution_tr" is the result for training set
  # "consistency_rate_distribution_te" is the result for test set
  consistency_rate_distribution_tr <- tibble(model = names(mt_trs),
                                             consistency_rate_distribution = vector("list", 1))
  consistency_rate_distribution_te <- tibble(model = names(mt_tes),
                                             consistency_rate_distribution = vector("list", 1))
  
  for (j in 1:length(mt_trs)) {
    mt_te <- mt_tes[[j]]
    mt_tr <- mt_trs[[j]]
    
    consistency_rate_distribution_te$consistency_rate_distribution[[j]] <-
      get_consistency_rate_per_event(mt_te)
    consistency_rate_distribution_tr$consistency_rate_distribution[[j]] <-
      get_consistency_rate_per_event(mt_tr)
  }
  
  consistency_rate_distribution_te <- consistency_rate_distribution_te %>%
    unnest(consistency_rate_distribution) %>%
    mutate(case = "Test set")
  
  consistency_rate_distribution_tr <- consistency_rate_distribution_tr %>%
    unnest(consistency_rate_distribution) %>%
    mutate(case = "Training set")
  
  eval_grid$consistency_rate_distribution[[i]] <- consistency_rate_distribution_te %>%
    rbind(consistency_rate_distribution_tr)
}

# prepare data for piloting

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

# combine all the result to a single table
data_plot <- eval_grid %>%
  select(region, season, iter, consistency_rate_distribution) %>%
  unnest(consistency_rate_distribution) %>%
  mutate(model = factor(model, levels = model_order))

# Plot summer floods ------------------------------------------------------

data_plot2 <- data_plot %>%
  dplyr::filter(region %in% c(1:4),
                season %in% c("S"))%>%
  group_by(region, season, model) %>%
  mutate(region = paste0("Region ", region))

# K-S test
# split the tibble of all results to sub-tibbles of each outer CV iteration
data_plot_dist_test <- data_plot2 %>%
  group_by(region, season, model) %>%
  group_split()

# create a tibble to store the assessment result
data_plot_dist_test_summary <-  data_plot_dist_test %>%
  lapply(function(x)
    x[1,]) %>%
  bind_rows() %>%
  mutate(test_result = "")

# perform two-sample KS test; iterate over the outer CV iterations
for (i in seq_along(data_plot_dist_test)) {
  data_plot_dist_test_summary$test_result[i] <-
    data_plot_dist_test[[i]] %>%
    dist_test_df()
}

# histgram plot + text label
ggplot(data_plot2, aes(consistency_rate_distribution, fill = case)) +
  geom_histogram(
    aes(y = 0.05 * ..density..),
    alpha = 0.5,
    position = 'identity',
    binwidth = 0.05,
    color = "grey30",
    size = 0.05
  ) +
  geom_text(
    data = data_plot_dist_test_summary,
    aes(
      x = 0,
      y = 0.8,
      label = test_result,
      color = factor(test_result)
    ),
    size = 3.2,
    hjust = 0,
    vjust = 0.5,
    parse = F
  ) +
  scale_color_manual(values = c("midnightblue", "indianred4"),
                     guide = "none") +
  scale_y_continuous(breaks = c(0, 0.5, 1)) +
  facet_grid(model ~ region) +
  labs(
    fill = "Associated\ndataset",
    x = expression(Consistency ~ rate ~ with ~ respect ~ to ~ MR[1]),
    y = "Proportion",
    tag = "H0: distributions of the consistency rate of the inputs \n associated with training and test sets are the same."
  ) +
  theme_bw(base_size = 10) +
  theme(
    legend.position = 'top',
    legend.justification = 'left',
    legend.direction = 'horizontal',
    strip.background = element_rect(fill = "grey80", size = 0),
    strip.text.y = element_text(size = 7),
    panel.spacing.x = unit(0.3, "lines"),
    panel.spacing.y = unit(0.4, "lines"),
    panel.border = element_blank(),
    axis.text.x = element_text(angle = 90, size = 9),
    axis.text.y = element_text(size = 9),
    plot.tag.position = c(0.7, 0.975),
    plot.tag = element_text(size = 9)
  )

ggsave(
  filename = "./paper_figures/Figure7.png",
  width = 7,
  height = 8.5,
  units = "in",
  dpi = 600
)





# Plot winter floods ------------------------------------------------------

data_plot2 <- data_plot %>%
  dplyr::filter(region %in% c(1:4),
                season %in% c("W"))%>%
  group_by(region, season, model) %>%
  mutate(region = paste0("Region ", region))

# K-S test
# split the tibble of all results to sub-tibbles of each outer CV iteration
data_plot_dist_test <- data_plot2 %>%
  group_by(region, season, model) %>%
  group_split()

# create a tibble to store the assessment result
data_plot_dist_test_summary <-  data_plot_dist_test %>%
  lapply(function(x)
    x[1,]) %>%
  bind_rows() %>%
  mutate(test_result = "")

# perform two-sample KS test; iterate over the outer CV iterations
for (i in seq_along(data_plot_dist_test)) {
  data_plot_dist_test_summary$test_result[i] <-
    data_plot_dist_test[[i]] %>%
    dist_test_df()
}

# histgram plot + text label
ggplot(data_plot2, aes(consistency_rate_distribution, fill = case)) +
  geom_histogram(
    aes(y = 0.05 * ..density..),
    alpha = 0.5,
    position = 'identity',
    binwidth = 0.05,
    color = "grey30",
    size = 0.05
  ) +
  geom_text(
    data = data_plot_dist_test_summary,
    aes(
      x = 0,
      y = 0.8,
      label = test_result,
      color = factor(test_result)
    ),
    size = 3.2,
    hjust = 0,
    vjust = 0.5,
    parse = F
  ) +
  scale_color_manual(values = c("midnightblue", "indianred4"),
                     guide = "none") +
  scale_y_continuous(breaks = c(0, 0.5, 1)) +
  facet_grid(model ~ region) +
  labs(
    fill = "Associated\ndataset",
    x = expression(Consistency ~ rate ~ with ~ respect ~ to ~ MR[1]),
    y = "Proportion",
    tag = "H0: distributions of the consistency rate of the inputs \n associated with training and test sets are the same."
  ) +
  theme_bw(base_size = 10) +
  theme(
    legend.position = 'top',
    legend.justification = 'left',
    legend.direction = 'horizontal',
    strip.background = element_rect(fill = "grey80", size = 0),
    strip.text.y = element_text(size = 7),
    panel.spacing.x = unit(0.3, "lines"),
    panel.spacing.y = unit(0.4, "lines"),
    panel.border = element_blank(),
    axis.text.x = element_text(angle = 90, size = 9),
    axis.text.y = element_text(size = 9),
    plot.tag.position = c(0.7, 0.975),
    plot.tag = element_text(size = 9)
  )

ggsave(
  filename = "./paper_figures/Figure_S6.png",
  width = 7,
  height = 8.5,
  units = "in",
  dpi = 600
)

