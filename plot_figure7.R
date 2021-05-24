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


# Constant ----------------------------------------------------------------

old_model_names <-
  c(
    "lmFit",
    "MARSFit"  ,
    "PLSFit"   ,
    "RidgeFit"    ,
    "LassoFit",
    "CARTFit"  ,
    "KNNFit",
    "CubistFit"  ,
    "SVMFit"    ,
    "svmRadialFit" ,
    "RFFit"       ,
    "XGBoost"
  )

new_model_names <-
  c(
    "LM",
    "MARSBag",
    "PLS",
    "Ridge",
    "Lasso",
    "CART",
    "KNN",
    "Cubist",
    "SVMPoly",
    "SVMRadial",
    "RF",
    "XGBoost"
  )


# Data --------------------------------------------------------------------

load("./mt_results/rain_mt.Rda")


# Plot --------------------------------------------------------------------


get_consistent_rate <- function(x) {
  (sum(x == 4)) / length(x)
}

get_consistent_rate_per_event <- function(xs) {
  m <-
    xs %>% unlist() %>% matrix(nrow = length(xs)) # num of mt * num of flood events
  
  apply(m, 2, get_consistent_rate)
}

eval_grid <- eval_grid %>%
  mutate(consistent_dis = vector("list", 1))

# iterate over region, season, data splits
for (i in 1:nrow(eval_grid)) {
  mt_trs <- eval_grid$mt_trs[[i]]
  mt_tes <- eval_grid$mt_tes[[i]]
  
  # iteration over machine learning methods
  consistent_dis_tr <- tibble(model = names(mt_trs),
                              consistent_dis = vector("list", 1))
  consistent_dis_te <- consistent_dis_tr
  
  for (j in 1:length(mt_trs)) {
    mt_te <- mt_tes[[j]]
    mt_tr <- mt_trs[[j]]
    
    consistent_dis_te$consistent_dis[[j]] <-
      get_consistent_rate_per_event(mt_te)
    consistent_dis_tr$consistent_dis[[j]] <-
      get_consistent_rate_per_event(mt_tr)
  }
  
  consistent_dis_te <- consistent_dis_te %>%
    unnest(consistent_dis) %>%
    mutate(case = "Test set")
  
  consistent_dis_tr <- consistent_dis_tr %>%
    unnest(consistent_dis) %>%
    mutate(case = "Training set")
  
  eval_grid$consistent_dis[[i]] <- consistent_dis_te %>%
    rbind(consistent_dis_tr)
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
  select(region, season, iter, consistent_dis) %>%
  unnest(consistent_dis)

data_plot <- data_plot %>%
  mutate(model = factor(model, levels = model_order, labels =  replace(model_order, model_order == "CART", "CARTBag")))


# plot summer

data_plot2 <- data_plot %>%
  dplyr::filter(region %in% c(1:4),
                season %in% c("S"))%>%
  group_by(region, season, model) %>%
  mutate(region = paste0("Region ", region))

# K-S test

dist_test <- function(data_test1,
                      data_test2,
                      confidence_level = 0.05) {
  test_result <- cvm_test(data_test1, data_test2)
  
  if (test_result[2] <= confidence_level) {
    "reject H[0]"
  } else {
    "fail to reject H[0]"
  }
}

dist_test_df <- function(df) {
  data_test1 <- df %>%
    dplyr::filter(case == "Test set") %>%
    pull(consistent_dis)
  
  data_test2 <- df %>%
    dplyr::filter(case == "Training set") %>%
    pull(consistent_dis)
  
  dist_test(data_test1, data_test2)
}

data_plot_dist_test <- data_plot2 %>%
  group_by(region, season, model) %>%
  group_split()
data_plot_dist_test_summary <-  data_plot_dist_test %>%
  lapply(function(x)
    x[1, ]) %>%
  bind_rows() %>%
  mutate(test_result = "")

for (i in seq_along(data_plot_dist_test)) {
  data_plot_dist_test_summary$test_result[i] <-
    data_plot_dist_test[[i]] %>%
    dist_test_df()
}

data_plot_dist_test_summary2 <- data_plot_dist_test_summary %>%
  mutate(test_result = replace(test_result, test_result == "reject H[0]", "reject H0")) %>%
  mutate(test_result = replace(
    test_result,
    test_result == "fail to reject H[0]",
    "fail to reject H0"
  ))

ggplot(data_plot2, aes(consistent_dis, fill = case)) +
  geom_histogram(
    aes(y = 0.05 * ..density..),
    alpha = 0.5,
    position = 'identity',
    binwidth = 0.05,
    color = "grey30",
    size = 0.05
  ) +
  geom_text(
    data = data_plot_dist_test_summary2,
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
                     guide = F) +
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

 ggsave(
   filename = "./paper_figures/Figure7.pdf",
   width = 7,
   height = 8.5,
   units = "in"
 )
 
 
 
 # plot winter
 
 data_plot2 <- data_plot %>%
   dplyr::filter(region %in% c(1:4),
                 season %in% c("W"))%>%
   group_by(region, season, model) %>%
   mutate(region = paste0("Region ", region))
 
 # K-S test
 dist_test <- function(data_test1,
                       data_test2,
                       confidence_level = 0.05) {
   test_result <- cvm_test(data_test1, data_test2)
   
   if (test_result[2] <= confidence_level) {
     "reject H[0]"
   } else {
     "fail to reject H[0]"
   }
 }
 
 dist_test_df <- function(df) {
   data_test1 <- df %>%
     dplyr::filter(case == "Test set") %>%
     pull(consistent_dis)
   
   data_test2 <- df %>%
     dplyr::filter(case == "Training set") %>%
     pull(consistent_dis)
   
   dist_test(data_test1, data_test2)
 }
 
 data_plot_dist_test <- data_plot2 %>%
   group_by(region, season, model) %>%
   group_split()
 data_plot_dist_test_summary <-  data_plot_dist_test %>%
   lapply(function(x)
     x[1, ]) %>%
   bind_rows() %>%
   mutate(test_result = "")
 
 for (i in seq_along(data_plot_dist_test)) {
   data_plot_dist_test_summary$test_result[i] <-
     data_plot_dist_test[[i]] %>%
     dist_test_df()
 }
 
 data_plot_dist_test_summary2 <- data_plot_dist_test_summary %>%
   mutate(test_result = replace(test_result, test_result == "reject H[0]", "reject H0")) %>%
   mutate(test_result = replace(
     test_result,
     test_result == "fail to reject H[0]",
     "fail to reject H0"
   ))
 
 ggplot(data_plot2, aes(consistent_dis, fill = case)) +
   geom_histogram(
     aes(y = 0.05 * ..density..),
     alpha = 0.5,
     position = 'identity',
     binwidth = 0.05,
     color = "grey30",
     size = 0.05
   ) +
   geom_text(
     data = data_plot_dist_test_summary2,
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
                      guide = F) +
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
 
 ggsave(
   filename = "./paper_figures/Figure_S6.pdf",
   width = 7,
   height = 8.5,
   units = "in"
 )