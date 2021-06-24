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


# MR1 ---------------------------------------------------------------------

# "mr1_mt.Rda" stores the results of experiments with MR1

load("./mt_results/mr1_mt.Rda")

# extract goodness-of-fit (gof) results from "eval_grid"
data_gof <- eval_grid %>%
  select(region, season, iter, gof_result) %>%
  unnest(gof_result)

# order model according to r2
model_order <- data_gof %>%
  group_by(model) %>%
  dplyr::summarise(mean_gof = mean(r2)) %>%
  arrange(desc(mean_gof)) %>%
  pull(model)

# function to compute consistency rate; the consistent assessment results are labelled with 4.
get_consistency_rate <- function(x) {
  (sum(x == 4)) / length(x)
}

# The assessments of the experiment is stored in "eval_grid"
# consistency_rate_tes: consistency rate for test sets
eval_grid <- eval_grid %>%
  mutate(consistency_rate_tes = vector("list", 1))

# iterate over region, season, data splits
for (i in 1:nrow(eval_grid)) {
  mt_tes <- eval_grid$mt_tes[[i]]
  
  # iteration over machine learning methods
  # "consistency_rate_te" stores the consistency rate of each ML method for one row of eval_grid;
  # -9999 is a placeholder
  consistency_rate_te <- tibble(model = names(mt_tes),
                                consistency_rate = -9999)
  
  # iterate over the models
  for (j in 1:length(mt_tes)) {
    consistency_rate_te$consistency_rate[[j]] <- sapply(mt_tes[[j]], get_consistency_rate) %>% # a rate for each change magnitude
      mean() # compute the mean of all change magnitudes
  }
  
  # store the result for one row of "eval_grid"
  eval_grid$consistency_rate_tes[[i]] <- consistency_rate_te
}

# rename "eval_grid" to "eval_grid1" as it stores the results with respect to MR1 
eval_grid1 <- eval_grid %>%
  select(region, season, iter, consistency_rate_tes) %>%
  unnest(cols = consistency_rate_tes)


# MR3 ---------------------------------------------------------------------

# "mr3_mt.Rda" stores the results of experiments with MR1;
# The consistency results for MR3 of each model have already been computed
load("./mt_results/mr3_mt.Rda")

eval_grid3 <- eval_grid %>%
  dplyr::select(region, season, iter, mt_tes) %>%
  unnest(cols = mt_tes)

# Plot --------------------------------------------------------------------

# join the results of MR1 and MR3 and the GOF results
data_plot <- eval_grid3 %>%
  left_join(eval_grid1, by = c("region", "season", "iter", "model")) %>%
  left_join(data_gof, by = c("region", "season", "iter", "model")) %>%
  dplyr::rename(mr3 = consistency_rate.x,
                mr1 = consistency_rate.y,
                r2 = r2) %>%
  gather(item, value, mr1, r2) %>% # note results for "MR3" is not joint; it is used as the axis of the plot
  mutate(
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
    item = factor(
      item,
      levels = c("r2", "mr1")
    ),
    model = factor(model, levels = model_order)
  )

# labels of the facet strips
my_labeller <-
  as_labeller(
    c(
      r2 = "R^2",
      mr1 = "Consistency~rate~with~respect~MR[1]",
      `Region 1` = "Region~1",
      `Region 2` = "Region~2",
      `Region 3` = "Region~3",
      `Region 4` = "Region~4"
    ),
    default = label_parsed
  )

# The error bars shown on the plot
data_plot2 <- data_plot %>%
  group_by(region, season, model, item) %>%
  dplyr::summarise(
    mean_value = mean(value),
    min_value = min(value),
    max_value = max(value),
    mean_mr3 = mean(mr3),
    max_mr3 = max(mr3),
    min_mr3 = min(mr3),
    .groups = 'drop'
  )

# plot only the summer floods
ggplot() +
  geom_point(
    data = data_plot %>% filter(season == "Summer"),
    aes(x = mr3, y = value, color = model),
    size = 1,
    alpha = 0.5
  ) +
  geom_errorbar( # the vertical error bars
    data = data_plot2 %>% filter(season == "Summer"),
    aes(
      x = mean_mr3,
      y = mean_value,
      ymin = min_value,
      ymax = max_value,
      color = model
    ),
    size = 0.5
  ) +
  geom_errorbar( # the horizontal error bars
    data = data_plot2 %>% filter(season == "Summer"),
    aes(
      x = mean_mr3,
      y = mean_value,
      xmin = min_mr3,
      xmax = max_mr3,
      color = model
    ),
    size = 0.5
  ) +
  geom_text_repel( # labels the model using text
    data = data_plot2 %>% filter(season == "Summer"),
    aes(
      label = model %>% as.character(),
      x = mean_mr3,
      y = mean_value,
    ),
    max.iter = 1000000,
    force = 10,
    xlim = c(0.2, 1),
    size = 1.8,
    segment.size = 0.09,
    max.overlaps = 14
  ) +
  scale_color_discrete() +
  scale_y_continuous(limits = c(0.24, 1.05),
                     breaks = c(0.2, 0.4, 0.6, 0.8, 1)) +
  scale_x_continuous(limits = c(0.6, 1),
                     breaks = c(0.6, 0.7, 0.8, 0.9, 1)) +
  facet_grid(item ~ region, switch = "y", labeller = my_labeller)+
  labs(
    x = expression(Consistency ~ rate ~ with ~ respect ~ to ~ MR[3]),
    color = "Model"
  ) +
  theme_bw(base_size = 10)+
  theme(legend.position = "right",
        strip.background.x = element_rect(fill = "grey80", size = 0),
        strip.background.y = element_blank(),
        axis.title.y = element_blank(),
        axis.title.x = element_text(size = 9),
        strip.placement = "outside",
        )


ggsave(
  filename = "./paper_figures/Figure10.png",
  width = 7,
  height = 5.5,
  units = "in",
  dpi = 600
)


# Plot winter -------------------------------------------------------------

ggplot() +
  geom_point(
    data = data_plot %>% filter(season == "Winter"),
    aes(x = mr3, y = value, color = model),
    size = 1,
    alpha = 0.5
  ) +
  geom_errorbar(
    data = data_plot2 %>% filter(season == "Winter"),
    aes(
      x = mean_mr3,
      y = mean_value,
      ymin = min_value,
      ymax = max_value,
      color = model
    ),
    size = 0.5
  ) +
  geom_errorbar(
    data = data_plot2 %>% filter(season == "Winter"),
    aes(
      x = mean_mr3,
      y = mean_value,
      xmin = min_mr3,
      xmax = max_mr3,
      color = model
    ),
    size = 0.5
  ) +
  geom_text_repel(
    data = data_plot2 %>% filter(season == "Winter"),
    aes(
      label = model %>% as.character(),
      x = mean_mr3,
      y = mean_value,
    ),
    max.iter = 1000000,
    force = 10,
    xlim = c(0.2, 1),
    size = 1.8,
    segment.size = 0.09,
    max.overlaps = 14
  ) +
  scale_color_discrete() +
  scale_y_continuous(limits = c(0.24, 1.05),
                     breaks = c(0.2, 0.4, 0.6, 0.8, 1)) +
  scale_x_continuous(limits = c(0.6, 1),
                     breaks = c(0.6, 0.7, 0.8, 0.9, 1)) +
  facet_grid(item ~ region, switch = "y", labeller = my_labeller)+
  labs(
    x = expression(Consistency ~ rate ~ with ~ respect ~ to ~ MR[3]),
    color = "Model"
  ) +
  theme_bw(base_size = 10)+
  theme(legend.position = "right",
        strip.background.x = element_rect(fill = "grey80", size = 0),
        strip.background.y = element_blank(),
        axis.title.y = element_blank(),
        axis.title.x = element_text(size = 9),
        strip.placement = "outside",
  )


ggsave(
  filename = "./paper_figures/Figure_S9.png",
  width = 7,
  height = 5.5,
  units = "in",
  dpi = 600
)


