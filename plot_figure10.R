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
  twosamples,
  scales
)


# Rain MT -----------------------------------------------------------------

load("./mt_results/rain_mt.Rda")

# GOF
data_gof <- eval_grid %>%
  select(region, season, iter, gof_result) %>%
  unnest(gof_result)

model_order <- data_gof %>%
  group_by(model) %>%
  dplyr::summarise(mean_gof = mean(r2)) %>%
  arrange(desc(mean_gof)) %>%
  pull(model)

# MR1
get_consistent_rate <- function(x) {
  (sum(x == 4)) / length(x)
}

eval_grid <- eval_grid %>%
  mutate(consistent_rate_tes = vector("list", 1))

# iterate over region, season, data splits
for (i in 1:nrow(eval_grid)) {
  mt_tes <- eval_grid$mt_tes[[i]]
  
  # iteration over machine learning methods
  consistent_rate_te <- tibble(model = names(mt_tes),
                               consistent_rate = -9999)
  
  for (j in 1:length(mt_tes)) {
    consistent_rate_te$consistent_rate[[j]] <- sapply(mt_tes[[j]], get_consistent_rate) %>% # a rate for each change magnitude
      mean()# compute the mean of all change magnitudes
  }
  
  eval_grid$consistent_rate_tes[[i]] <- consistent_rate_te
}

eval_grid1 <- eval_grid %>%
  select(region, season, iter, consistent_rate_tes) %>%
  unnest(cols = consistent_rate_tes) 

# MR3 ---------------------------------------------------------------------

load("./mt_results/mr3_mt.Rda")

eval_grid3 <- eval_grid %>%
  dplyr::select(region, season, iter, org_mt) %>%
  unnest(cols = org_mt)

# Plot --------------------------------------------------------------------

data_plot <- eval_grid3 %>%
  left_join(eval_grid1, by = c("region", "season", "iter", "model")) %>%
  left_join(data_gof, by = c("region", "season", "iter", "model")) %>%
  dplyr::rename(mr3 = org_consistent_rate,
                mr1 = consistent_rate,
                r2 = r2) %>%
  gather(item, value, mr1, r2) %>%
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
    model = factor(model, levels = model_order, labels =  replace(model_order, model_order == "CART", "CARTBag"))
  )


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

ggplot() +
  geom_point(
    data = data_plot %>% filter(season == "Summer"),
    aes(x = mr3, y = value, color = model),
    size = 1,
    alpha = 0.5
  ) +
  geom_errorbar(
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
  geom_errorbar(
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
  geom_text_repel(
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
  filename = "./paper_figures/figure10.png",
  width = 7,
  height = 5.5,
  units = "in",
  dpi = 600
)

ggsave(
  filename = "./paper_figures/figure10.pdf",
  width = 7,
  height = 5.5,
  units = "in"
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
  filename = "./paper_figures/figure_S9.png",
  width = 7,
  height = 5.5,
  units = "in",
  dpi = 600
)

ggsave(
  filename = "./paper_figures/figure_S9.pdf",
  width = 7,
  height = 5.5,
  units = "in"
)