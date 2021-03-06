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
  lemon,
  ggrepel,
  scales
)


# data --------------------------------------------------------------------

# load the assessment results with respect to each MR
load("./mt_results/mr1_mt.Rda")
p_eval_grid <- eval_grid

load("./mt_results/mr2_mt.Rda")
pet_eval_grid <- eval_grid

load("./mt_results/mr1_mr2_mt.Rda")
p_pet_eval_grid <- eval_grid


# # "data_gof" save the goodness of fit results for ranking the models;
# the results are stored in "model_order"
data_gof <- eval_grid %>%
  select(region, season, iter, gof_result) %>%
  unnest(gof_result)

model_order <- data_gof %>%
  group_by(model) %>%
  dplyr::summarise(mean_gof = mean(r2)) %>%
  arrange(desc(mean_gof)) %>%
  pull(model)


# pre-processing ----------------------------------------------------------

# The assessment results are labeled with numbers; consistent results are labeled with 4
get_consistency_rate <- function(x) {
  (sum(x == 4)) / length(x)
}

# if both assessment are consistent,then the sum of the two assessments are 8
get_consistency_rate_p_pet <- function(x1, x2) {
  x <- x1+x2
  (sum(x == 8)) / length(x)
}

# Process the data for MR1
#
# create a column to store the average consistency rate for an outer CV iteration
p_eval_grid <- p_eval_grid %>%
  mutate(test_results = vector("list", 1))

# iterate over region, season, models
for (i in 1:nrow(p_eval_grid)) {
  # extract assessment results associated with the test set
  mt_tes <- p_eval_grid$mt_tes[[i]]
  
  # create a tibble to store the result for different models
  test_results <- tibble(model = names(mt_tes),
                         test_result = -9999)
  
  # iteration over machine learning methods
  for (j in seq_along(mt_tes)) {
    test_results$test_result[[j]] <-
      mt_tes[[j]] %>% unlist() %>% get_consistency_rate()
  }
  
  # store the store for the current iteration
  p_eval_grid$test_results[[i]] <- test_results
}

# combine all the result into a tibble "p"
p <- p_eval_grid %>%
  select(region, season, iter, test_results) %>%
  unnest(test_results) %>%
  dplyr::rename(mr1 = test_result)


# Process the data for MR2
#
# create a column to store the average consistency rate for an outer CV iteration
pet_eval_grid <- pet_eval_grid %>%
  mutate(test_results = vector("list", 1))

# iterate over region, season, models
for (i in 1:nrow(pet_eval_grid)) {
  # extract assessment results associated with the test set
  mt_tes <- pet_eval_grid$mt_tes[[i]]
  
  # create a tibble to store the result for different models
  test_results <- tibble(model = names(mt_tes),
                         test_result = -9999)
  
  # iteration over machine learning methods
  for (j in seq_along(mt_tes)) {
    test_results$test_result[[j]] <-
      mt_tes[[j]] %>% unlist() %>% get_consistency_rate()
  }
  
  # store the store for the current iteration
  pet_eval_grid$test_results[[i]] <- test_results
}

# combine all the result into a tibble "pet"
pet <- pet_eval_grid %>%
  select(region, season, iter, test_results) %>%
  unnest(test_results) %>%
  dplyr::rename(mr2 = test_result)

# Process the data for assessments using both MR1 and MR2
#
# create a column to store the average consistency rate for an outer CV iteration
p_pet_grid <- p_pet_eval_grid %>%
  mutate(test_results = vector("list", 1))

# iterate over region, season, models
for (i in 1:nrow(p_pet_eval_grid)) {
  # extract assessment results associated with the test set
  # "mt_tes_p_pet" stores the results when both MR1 and MR2 are used in the assessment; 
  #  MR3 is used to further modify the inputs, which have already been modified by MR1
  # "mt_tes_p" stores the results when MR1 is used in assessment 
  mt_tes_p_pet <- p_pet_eval_grid$mt_tes[[i]]
  mt_tes_p <- p_eval_grid$mt_tes[[i]]
  
  # create a tibble to store the result for different models
  test_results <- tibble(model = names(mt_tes),
                         test_result = -9999)
  
  # iteration over machine learning methods
  for (j in seq_along(mt_tes_p_pet)) {
    x1 <- mt_tes_p_pet[[j]] %>% unlist()
    x2 <- mt_tes_p[[j]] %>% unlist()
    
    test_results$test_result[[j]] <- get_consistency_rate_p_pet(x1, x2)
  }
  
  # store the store for the current iteration
  p_pet_grid$test_results[[i]] <- test_results
}

# combine all the result into a tibble "p_pet"
p_pet <- p_pet_grid %>%
  select(region, season, iter, test_results) %>%
  unnest(test_results) %>%
  dplyr::rename(mr1_mr2 = test_result)


# Plot --------------------------------------------------------------------

# Join the assessment results from all three assessment experiments;
# format the labels, and arranging the results
data_plot <- p %>%
  left_join(pet,  by = c("region", "season", "iter", "model")) %>%
  left_join(p_pet,  by = c("region", "season", "iter", "model")) %>%
  gather(
    item, value, mr1, mr2, mr1_mr2
  ) %>%
  dplyr::mutate(
    item = factor(
      item,
      levels = c("mr1", "mr2", "mr1_mr2"),
      labels = c("MR[1]", "MR[2]", "MR[1]*' & M'*R[2]")
    ),
    region = factor(
      region,
      levels = c(1:4),
      labels = str_c("Region ", 1:4)
    ),
    season = factor(
      season,
      levels = c("S", "W"),
      labels = c("Summer", "Winter")
    ),
    model = factor(model, levels = model_order)
  )

# create the labs for piloting
lab <- c(expression(MR[1]),
         expression(MR[2]),
         expression(MR[1] * ' & M' * R[2]))



# Plotting the results for summer floods ----------------------------------

ggplot(data_plot %>% filter(season == "Summer"), aes(item, value)) +
  geom_point(aes(color = item, shape = item),
             size = 1,
             stroke = 0.3) +
  geom_line(aes(group = interaction(iter, season)), size = 0.2, color =
              "grey50") +
  scale_color_manual(
    name = "MR considered",
    values = c("#00AFBB", "#E7B800", "#FC4E07"),
    labels = lab
  ) +
  scale_shape_discrete(name = "MR considered",
                       solid = F,
                       labels = lab) +
  scale_x_discrete(labels = parse(text = levels(data_plot$item))) +
  facet_grid(region ~ model) +
  labs(x = "MR considered",
       y = "Consistency rate") +
  theme_bw(base_size = 9) +
  theme(legend.position = "top") +    
  theme(
    legend.position = "top",
    strip.background = element_rect(fill = "grey80", size = 0, color = NA),
    strip.text = element_text(size = 7),
    panel.border = element_blank(),
    panel.spacing.x = unit(0.16, "lines"),
    axis.text.y =  element_text(size = 7),
    axis.text.x =  element_text(size = 7, angle = 90),
    panel.background = element_rect(fill = "grey95")
  )

ggsave(
  filename = "./paper_figures/Figure9.png",
  width = 7,
  height = 4.8,
  units = "in",
  dpi = 600
)


# Plotting results on winter floods ---------------------------------------

ggplot(data_plot %>% filter(season == "Winter"), aes(item, value)) +
  geom_point(aes(color = item, shape = item),
             size = 1,
             stroke = 0.3) +
  geom_line(aes(group = interaction(iter, season)), size = 0.2, color =
              "grey50") +
  scale_color_manual(
    name = "MR considered",
    values = c("#00AFBB", "#E7B800", "#FC4E07"),
    labels = lab
  ) +
  scale_shape_discrete(name = "MR considered",
                       solid = F,
                       labels = lab) +
  scale_x_discrete(labels = parse(text = levels(data_plot$item))) +
  facet_grid(region ~ model) +
  labs(x = "MR considered",
       y = "Consistency rate") +
  theme_bw(base_size = 9) +
  theme(legend.position = "top") +    
  theme(
    legend.position = "top",
    strip.background = element_rect(fill = "grey80", size = 0, color = NA),
    strip.text = element_text(size = 7),
    panel.border = element_blank(),
    panel.spacing.x = unit(0.16, "lines"),
    axis.text.y =  element_text(size = 7),
    axis.text.x =  element_text(size = 7, angle = 90),
    panel.background = element_rect(fill = "grey95")
  )

ggsave(
  filename = "./paper_figures/Figure_S8.png",
  width = 7,
  height = 4.8,
  units = "in",
  dpi = 600
)

