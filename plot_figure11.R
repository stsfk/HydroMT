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


# Load data original xgboost ----------------------------------------------


# load data
load("./mt_results/rain_mt.Rda")
eval_grid_mr1 <- eval_grid

load('./mt_results/pet_mt.Rda')
eval_grid_mr2 <- eval_grid

load("./mt_results/mr3_mt.Rda")
eval_grid_mr3 <- eval_grid

# get r2
data_r2 <- eval_grid_mr1 %>%
  select(region, season, iter, gof_result) %>%
  unnest(gof_result) %>%
  dplyr::filter(model == "XGBoost")


# get mr1
get_consistent_rate <- function(x) {
  (sum(x == 4)) / length(x)
}

eval_grid_mr1 <- eval_grid_mr1 %>%
  mutate(
    consistent_rate_tes = vector("list", 1)
  )

  # iterate over region, season, iters
for (i in 1:nrow(eval_grid_mr1)) {
  mt_tes <- eval_grid_mr1$mt_tes[[i]]
  
  eval_grid_mr1$consistent_rate_tes[[i]] <-
    tibble(model = "XGBoost",
           consistent_rate = mt_tes$XGBoost %>% unlist() %>% get_consistent_rate)
}

data_mr1 <- eval_grid_mr1 %>%
  select(region, season, iter, consistent_rate_tes) %>%
  unnest(cols = consistent_rate_tes) %>%
  dplyr::rename(mr1 = consistent_rate) 


# get mr2
eval_grid_mr2 <- eval_grid_mr2 %>%
  mutate(
    consistent_rate_tes = vector("list", 1)
  )

# iterate over region, season, iters
for (i in 1:nrow(eval_grid_mr2)) {
  mt_tes <- eval_grid_mr2$mt_tes[[i]]
  
  eval_grid_mr2$consistent_rate_tes[[i]] <-
    tibble(model = "XGBoost",
           consistent_rate = mt_tes$XGBoost %>% unlist() %>% get_consistent_rate)
}

data_mr2 <- eval_grid_mr2 %>%
  select(region, season, iter, consistent_rate_tes) %>%
  unnest(cols = consistent_rate_tes) %>%
  dplyr::rename(mr2 = consistent_rate) 


# get mr3
data_mr3 <- eval_grid_mr3 %>%
  dplyr::select(region, season, iter, org_mt) %>%
  dplyr::rename(mr3 = org_mt) %>%
  unnest(mr3) %>%
  dplyr::rename(mr3 = org_consistent_rate) %>%
  dplyr::filter(model == "XGBoost")

# combine

data_original <- data_r2 %>%
  left_join(data_mr1, by = c("region", "season", "iter", "model")) %>%
  left_join(data_mr2, by = c("region", "season", "iter", "model")) %>%
  left_join(data_mr3, by = c("region", "season", "iter", "model"))


# Mono_xgboost ------------------------------------------------------------

load("./mt_results/mono_mt.Rda")

# Plot --------------------------------------------------------------------

data_plot <- data_original %>%
  bind_rows(data_mono) %>%
  gather(metric, value, r2:mr3) %>%
  mutate(
    region = factor(
      region,
      levels = c(1:4),
      labels = c("Region~1",
                 "Region~2",
                 "Region~3",
                 "Region~4")
    ),
    season = factor(
      season,
      levels = c("S", "W"),
      labels = c("Summer", "Winter")
    ),
    model = factor(
      model,
      levels = c("XGBoost", "XGBoost_mono"),
      labels = c("XGBoost", "XGBoostMono")
    ),
    metric = factor(
      metric,
      levels = c("r2", "mr1", "mr2", "mr3"),
      labels = c("R^2",
                 "MR[1]",
                 "MR[2]",
                 "MR[3]")
    )
  )


ggplot(data_plot %>% filter(season == "Summer"), aes(model, value)) +
  geom_point(aes(color = model, shape = model),
             size = 1.4,
             stroke = 0.3) +
  scale_shape_discrete(solid = F) +
  geom_line(aes(group = interaction(iter, season)), size = 0.4, color =
              "grey50") +
  facet_grid(region ~ metric, labeller = label_parsed) +
  labs(x = "Model",
       y = "Consistency rate or prediction accuracy") +
  theme_bw(base_size = 10) +
  theme(
    legend.position = "top",
    axis.text.x =  element_text(angle = 0),
  )

ggsave(
  filename = "./paper_figures/figure11.png",
  width = 7,
  height = 5.5,
  units = "in",
  dpi = 600
)

ggsave(
  filename = "./paper_figures/figure11.pdf",
  width = 7,
  height = 5.5,
  units = "in"
)



# Plot supporting info ----------------------------------------------------

ggplot(data_plot %>% filter(season == "Winter"), aes(model, value)) +
  geom_point(aes(color = model, shape = model),
             size = 1.4,
             stroke = 0.3) +
  scale_shape_discrete(solid = F) +
  geom_line(aes(group = interaction(iter, season)), size = 0.4, color =
              "grey50") +
  facet_grid(region ~ metric, labeller = label_parsed) +
  labs(x = "Model",
       y = "Consistency rate or prediction accuracy") +
  theme_bw(base_size = 10) +
  theme(
    legend.position = "top",
    axis.text.x =  element_text(angle = 0),
  )

ggsave(
  filename = "./paper_figures/figure_S10.png",
  width = 7,
  height = 5.5,
  units = "in",
  dpi = 600
)

ggsave(
  filename = "./paper_figures/figure_S10.pdf",
  width = 7,
  height = 5.5,
  units = "in"
)


