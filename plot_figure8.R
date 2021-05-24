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
  twosamples
)


# Data --------------------------------------------------------------------

load("./mt_results/mr1_mt.Rda")

get_consistent_rate <- function(x) {
  (sum(x == 4)) / length(x)
}

get_inconsistent_rate <- function(x) {
  (sum(x == 3)) / length(x)
}

get_invalid_rate <- function(x) {
  (sum(x == 2)) / length(x)
}

get_inconclusive_rate <- function(x) {
  (sum(x == 1)) / length(x)
}

get_evaluation_df <- function(xs, change_ratio = (50:150) * 0.01) {
  tibble(
    change_ratio = change_ratio,
    Consistent = sapply(xs, get_consistent_rate),
    Inconsistent = sapply(xs, get_inconsistent_rate),
    Invalid = sapply(xs, get_invalid_rate),
    Inconclusive = sapply(xs, get_inconclusive_rate)
  ) %>%
    gather(outcome, value, -change_ratio)
}

eval_grid <- eval_grid %>%
  mutate(evaluation_tes = vector("list", 1),
         evaluation_trs = vector("list", 1))

# iterate over region, season, data splits
for (i in 1:nrow(eval_grid)) {
  mt_trs <- eval_grid$mt_trs[[i]]
  mt_tes <- eval_grid$mt_tes[[i]]
  
  # iteration over machine learning methods
  evaluation_tr <- tibble(model = names(mt_trs),
                          evaluation = vector("list", 1))
  evaluation_te <- evaluation_tr
  
  for (j in 1:length(mt_trs)) {
    mt_te <- mt_tes[[j]]
    mt_tr <- mt_trs[[j]]
    
    evaluation_te$evaluation[[j]] <- get_evaluation_df(mt_te)
    evaluation_tr$evaluation[[j]] <- get_evaluation_df(mt_tr)
  }
  
  eval_grid$evaluation_tes[[i]] <- evaluation_te %>%
    unnest(evaluation)
  eval_grid$evaluation_trs[[i]] <- evaluation_tr %>%
    unnest(evaluation)
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
  select(region, season, iter, evaluation_tes) %>%
  unnest(cols = evaluation_tes)

data_plot <- data_plot %>%
  mutate(outcome = factor(
    outcome,
    levels = c("Consistent", "Inconsistent", "Inconclusive", "Invalid"),
    labels = c("Consistent", "Inconsistent", "Inconclusive test", "Invalid")
  )) %>%
  mutate(model = factor(model, levels = model_order,  labels =  replace(model_order, model_order == "CART", "CARTBag")))


# Plot summer
data_plot2 <- data_plot %>%
  dplyr::filter(region %in% c(1:4),
                season %in% c("S"))%>%
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


# Plot winter
data_plot2 <- data_plot %>%
  dplyr::filter(region %in% c(1:4),
                season %in% c("W"))%>%
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


