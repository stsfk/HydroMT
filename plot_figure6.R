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

dir_path <- c("./paper_figures")

if (!dir.exists(dir_path)){
  dir.create(dir_path)
}


# Rain MT -----------------------------------------------------------------

load("./mt_results/mr1_mt.Rda")

# model's GOF and consistent rate

get_consistent_rate <- function(x) {
  (sum(x == 4)) / length(x)
}

eval_grid <- eval_grid %>%
  mutate(
    consistent_rate_tes = vector("list", 1),
    consistent_rate_trs = vector("list", 1)
  )

# iterate over region, season, data splits
for (i in 1:nrow(eval_grid)) {
  mt_trs <- eval_grid$mt_trs[[i]]
  mt_tes <- eval_grid$mt_tes[[i]]
  
  # iteration over machine learning methods
  consistent_rate_tr <- tibble(model = names(mt_trs),
                               consistent_rate = vector("list", 1))
  consistent_rate_te <- consistent_rate_tr
  
  for (j in 1:length(mt_trs)) {
    mt_te <- mt_tes[[j]]
    mt_tr <- mt_trs[[j]]
    
    consistent_rate_te$consistent_rate[[j]] <-
      sapply(mt_te, get_consistent_rate)
    consistent_rate_tr$consistent_rate[[j]] <-
      sapply(mt_tr, get_consistent_rate)
  }
  
  eval_grid$consistent_rate_tes[[i]] <- consistent_rate_te
  eval_grid$consistent_rate_trs[[i]] <- consistent_rate_tr
}


data_gof <- eval_grid %>%
  select(region, season, iter, gof_result) %>%
  unnest(gof_result)

model_order <- data_gof %>%
  group_by(model) %>%
  dplyr::summarise(mean_gof = mean(r2)) %>%
  arrange(desc(mean_gof)) %>%
  pull(model)

data_plot <- eval_grid %>%
  select(region, season, iter, consistent_rate_tes) %>%
  unnest(cols = consistent_rate_tes) %>%
  mutate(mean_consistent_rate = map_dbl(consistent_rate, function(x)
    unlist(mean(x)))) %>%
  left_join(data_gof, by = c("region", "season", "iter", "model")) %>%
  dplyr::select(-consistent_rate) %>%
  dplyr::rename(gof = r2,
                consistent_rate = mean_consistent_rate) %>%
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
    model = factor(
      model,
      levels = model_order
    )
  )

data_plot2 <- data_plot %>%
  group_by(region, season, model) %>%
  dplyr::summarise(
    mean_consistent_rate = mean(consistent_rate),
    min_consistent_rate = min(consistent_rate),
    max_consistent_rate = max(consistent_rate),
    mean_gof = mean(gof),
    max_gof = max(gof),
    min_gof = min(gof)
  )

data_plot3 <- data_plot2 #%>%

p <- ggplot() +
  geom_point(
    data = data_plot,
    aes(x = gof, y = consistent_rate, color = model),
    size = 1,
    alpha = 0.5
  ) +
  geom_errorbar(
    data = data_plot2,
    aes(
      x = mean_gof,
      y = mean_consistent_rate,
      ymin = min_consistent_rate,
      ymax = max_consistent_rate,
      color = model
    ),
    size = 0.5
  ) +
  geom_errorbar(
    data = data_plot2,
    aes(
      x = mean_gof,
      y = mean_consistent_rate,
      xmin = min_gof,
      xmax = max_gof,
      color = model
    ),
    size = 0.5
  ) +
  geom_text_repel(
    data = data_plot3,
    aes(
      label = data_plot3$model %>% as.character(),
      x = mean_gof,
      y = mean_consistent_rate,
    ),
    max.iter = 1000000,
    force = 100,
    xlim = c(0.2, 1),
    size = 2,
    segment.size = 0.09,
    max.overlaps = 14
  )+
  scale_x_continuous(limits = c(0.18, 1),
                     breaks = c(0.2, 0.4, 0.6, 0.8, 1))+
  scale_y_continuous(limits = c(0.35, 1.05),
                     breaks = c(0.4, 0.6, 0.8, 1)) +
  scale_color_discrete() +
  facet_grid(season ~ region)+
  labs(y = expression(Consistency~rate~with~respect~to~MR[1]),
       x = "R²",
       color = "Model") +
  theme_bw(base_size = 10)+
  theme(legend.position = "right",
        strip.background = element_rect(fill = "grey80", size = 0))

ggsave(
  p,
  filename = "./paper_figures/Figure6.png",
  width = 7,
  height = 5.5,
  units = "in",
  dpi = 600
)
