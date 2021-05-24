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
  ggrepel
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

eval_grid <- expand.grid(
  region = c(1:4),
  season = c("S", "W"),
  iter = c(1:5)
) %>%
  as_tibble() %>%
  mutate(
    file_name = paste0(
      "./modeling_results/Reg",
      region,
      "_",
      season,
      "_iter",
      iter,
      ".Rda"
    ),
    xgb_file_name = paste0(
      "./modeling_results/xgb_region",
      region,
      "_",
      season,
      "_iter",
      iter,
      ".model"
    )
  ) %>%
  mutate(gof_result = vector("list", 1))


for (i in 1:nrow(eval_grid)) {
  # gof of all models except xgboost
  load(eval_grid$file_name[i])
  
  df1 <-
    sapply(models, function(x)
      hydroGOF::gof(predict(x, dtest), dtest$Qmax, digits = 8)[17]) %>%
    as_tibble() %>%
    mutate(model = names(models))
  
  # gof of xgboost model
  xgb_dtest <-
    xgb.DMatrix(data = data.matrix(dtest %>% select(-Qmax)),
                label = dtest$Qmax)
  
  xgb_model <- xgboost::xgb.load(eval_grid$xgb_file_name[i])
  
  df2 <- tibble(model = "XGBoost",
                value = hydroGOF::gof(
                  sim = predict(xgb_model, xgb_dtest),
                  ob = dtest$Qmax,
                  digits = 8
                )[17])
  
  # return
  eval_grid$gof_result[[i]] <- bind_rows(df1, df2) %>%
    dplyr::rename(r2 = value) %>%
    dplyr::select(model, r2) %>%
    mutate(model = plyr::mapvalues(model, old_model_names, new_model_names))
}



# Original sample MT ------------------------------------------------------

eval_grid <- eval_grid %>%
  mutate(org_mt = vector("list", 1))

consist_rate <- function(model, dtest, xgboost_format = F){
  # Input: 
  # Output: consistent rate
  
  # number of test
  n_tests <- nrow(dtest) - 1
  
  # get observation and prediction
  observation <- dtest$Qmax
  
  if (xgboost_format){
    xgb_dtest <-
      xgb.DMatrix(data = data.matrix(dtest %>% select(-Qmax)),
                  label = dtest$Qmax)
    
    prediction <- predict(model, xgb_dtest)
  } else {
    prediction <- predict(model, dtest)
  }
  
  # get only valid predictions
  valid_index <- (prediction >= 0)
  observation <- observation[valid_index]
  prediction <- prediction[valid_index] 
  
  # compute the number of consistent predictions
  n_consistents <- vector("double", length(prediction))
  for (i in 1:length(prediction)){
    n_inconsisitent <- ((prediction[i] - prediction)*(observation[i] - observation) < 0) %>% sum() # compute n_inconsistent first
    n_consistents[i] <- length(prediction) - n_inconsisitent - 1 # total - inconsistent - self
  }
  
  # output, the average consistent rate of valid predictions 
  out <- (n_consistents/n_tests) %>%
    mean()
  
  # reduction factor, in cases that an invalid prediction is picked
  out*length(prediction)/(n_tests + 1)
}

for (i in 1:nrow(eval_grid)) {
  # gof of all models except xgboost
  load(eval_grid$file_name[i])
  
  df1 <- sapply(models, consist_rate, dtest, xgboost_format = F)
  
  # gof of xgboost model
  xgb_model <- xgboost::xgb.load(eval_grid$xgb_file_name[i])
  
  df2 <- consist_rate(xgb_model, dtest, xgboost_format=T)
  
  # assign result
  eval_grid$org_mt[[i]] <- tibble(
    model = c(names(df1), "XGBoost"),
    org_consistent_rate = c(df1, df2))%>%
    mutate(model = plyr::mapvalues(model, old_model_names, new_model_names))
}

save(eval_grid, file = "./mt_results/mr3_mt.Rda")

# Plot --------------------------------------------------------------------
data_gof <- eval_grid %>%
  select(region, season, iter, gof_result) %>%
  unnest(gof_result)

model_order <- data_gof %>%
  group_by(model) %>%
  dplyr::summarise(mean_gof = mean(r2)) %>%
  arrange(desc(mean_gof)) %>%
  pull(model)

data_plots <- vector("list", nrow(eval_grid))
for (i in 1:length(data_plots)){
  data_plots[[i]] <- eval_grid$gof_result[[i]] %>%
    left_join(eval_grid$org_mt[[i]],  by = "model") %>%
    mutate(region = eval_grid$region[[i]],
           season = eval_grid$season[[i]],
           iter = eval_grid$iter[[i]])
}

data_plot <- data_plots %>%
  bind_rows() %>%
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
    model = factor(model, levels = model_order)
  )


data_plot2 <- data_plot %>%
  group_by(region, season, model) %>%
  dplyr::summarise(
    mean_consistent_rate = mean(org_consistent_rate),
    min_consistent_rate = min(org_consistent_rate),
    max_consistent_rate = max(org_consistent_rate),
    mean_gof = mean(r2),
    max_gof = max(r2),
    min_gof = min(r2)
  )

data_plot3 <- data_plot2 





ggplot() +
  geom_point(
    data = data_plot,
    aes(x = r2, y = org_consistent_rate, color = model),
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
    max.iter = 500000,
    force = 100,
    xlim = c(0.2, 1),
    size = 2,
    segment.size = 0.18,
    max.overlaps = 12
  ) +
  scale_x_continuous(limits = c(0.2, 1),
                     breaks = c(0.2, 0.4, 0.6, 0.8, 1))+
  scale_color_discrete() +
  facet_grid(season ~ region)+
  labs(y = expression(Consistent~rate~(MR[3])),
       x = "R²",
       color = "Model") +
  theme_bw(base_size = 10)+
  theme(strip.background = element_rect(fill = "grey80", size = 0))

ggsave(
  filename = "./mt_results/GoF_vs_org_consistent_rate.png",
  width = 7,
  height = 5.5,
  units = "in",
  dpi = 600
)






