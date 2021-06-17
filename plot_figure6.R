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


# MR1 results -------------------------------------------------------------

# "mr1_mt.Rda" stores the results of experiments with MR1
load("./mt_results/mr1_mt.Rda")

# model's goodness-of-fit (gof) and consistent rate
get_consistency_rate <- function(x) {
  # Consistent assessment results are labeled with "4"
  (sum(x == 4)) / length(x)
}

# The assessments of the experiment is stored in "eval_grid"
# consistency_rate_tes: consistent rate for test sets
# consistency_rate_trs: consistent rate for training sets
eval_grid <- eval_grid %>%
  mutate(
    consistency_rate_tes = vector("list", 1),
    consistency_rate_trs = vector("list", 1)
  )

# Evaluate each row of "eval_grid"; iterate over region, season, data splits
for (i in 1:nrow(eval_grid)) {
  # The assessment results for training and test set are stored in "mt_trs" and "mt_tes".
  mt_trs <- eval_grid$mt_trs[[i]]
  mt_tes <- eval_grid$mt_tes[[i]]
  
  # iteration over machine learning methods
  # "consistency_rate_tr" and "consistency_rate_te" store the consistency rate of each ML method for on row of eval_grid
  consistency_rate_tr <- tibble(model = names(mt_trs),
                               consistency_rate = vector("list", 1))
  consistency_rate_te <- tibble(model = names(mt_tes),
                               consistency_rate = vector("list", 1))
  
  for (j in 1:length(mt_trs)) {
    # iterate over ML methods
    mt_te <- mt_tes[[j]]
    mt_tr <- mt_trs[[j]]
    
    consistency_rate_te$consistency_rate[[j]] <-
      sapply(mt_te, get_consistency_rate)
    consistency_rate_tr$consistency_rate[[j]] <-
      sapply(mt_tr, get_consistency_rate)
  }
  
  eval_grid$consistency_rate_tes[[i]] <- consistency_rate_te
  eval_grid$consistency_rate_trs[[i]] <- consistency_rate_tr
}

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

# "data_plot": compute the mean consistency rate of the all experiment with respect to MR1 associated with test sets;
# join "data_gof", which stores the gof results of each model on the test sets;
# ordering and labeling the variables for plotting.
data_plot <- eval_grid %>%
  select(region, season, iter, consistency_rate_tes) %>%
  unnest(cols = consistency_rate_tes) %>%
  mutate(mean_consistency_rate = map_dbl(consistency_rate, function(x)
    unlist(mean(x)))) %>%
  left_join(data_gof, by = c("region", "season", "iter", "model")) %>%
  dplyr::select(-consistency_rate) %>% # drop the column that stores the detailed results; "mean_consistency_rate" is enough for plotting
  dplyr::rename(gof = r2,
                consistency_rate = mean_consistency_rate) %>%
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

# "data_plot2" stores the results on the ranges of the consistency_rate and gof; for making the lines/bar on the plot 
data_plot2 <- data_plot %>%
  group_by(region, season, model) %>%
  dplyr::summarise(
    mean_consistency_rate = mean(consistency_rate),
    min_consistency_rate = min(consistency_rate),
    max_consistency_rate = max(consistency_rate),
    mean_gof = mean(gof),
    max_gof = max(gof),
    min_gof = min(gof)
  )

# Making the plot using ggplot2; gof vs consistency rate with respect to MR1
# show the results of each model for each region, season, outer CV iteration using points;
# the average results are shown using bars and lines;
# for easy indentification, the models are labeled using text
p <- ggplot() +
  geom_point(
    data = data_plot,
    aes(x = gof, y = consistency_rate, color = model),
    size = 1,
    alpha = 0.5
  ) + # the vertical bars
  geom_errorbar(
    data = data_plot2,
    aes(
      x = mean_gof,
      y = mean_consistency_rate,
      ymin = min_consistency_rate,
      ymax = max_consistency_rate,
      color = model
    ),
    size = 0.5
  ) + # the horizontal lines
  geom_errorbar(
    data = data_plot2,
    aes(
      x = mean_gof,
      y = mean_consistency_rate,
      xmin = min_gof,
      xmax = max_gof,
      color = model
    ),
    size = 0.5
  ) + # to avoid overlap, the positions of the text labels are estimated multiple times ("max.iter")
  geom_text_repel(
    data = data_plot2,
    aes(
      label = data_plot2$model %>% as.character(),
      x = mean_gof,
      y = mean_consistency_rate,
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

# save the plot to file
ggsave(
  p,
  filename = "./paper_figures/Figure6.png",
  width = 7,
  height = 5.5,
  units = "in",
  dpi = 600
)
