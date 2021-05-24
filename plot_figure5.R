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
  xgboost,
  lemon,
  scales
)


random_figure <- function(ii,
                          row_of_interest = 17,
                          id_of_interested_event = 19) {
  # Predictions under changing rainfalls ------------------------------------
  
  # Function
  prep_data_mutation <- function(df,
                                 col_names,
                                 change_ratios = 1,
                                 change_amounts = 0) {
    # Change the value of col_names such that df[col_names] = df[col_names]*change_ratios +  change_amounts
    
    # In cases change_ratios is a single number, expand to same length as col_names
    
    change_ratios <- change_ratios * rep(1, length(col_names))
    change_amounts <- change_amounts * rep(1, length(col_names))
    
    for (i in seq_along(col_names)) {
      df[col_names[i]] <-
        df[col_names[i]] * change_ratios[i] + change_amounts[i]
    }
    
    df
  }
  
  # constant
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
      "RFFit"
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
      "RF"
    )
  
  col_names <- c("P0", "P1", "P3", "P5", "P7", "Pmin", "Pmax")
  
  # experiments
  
  eval_grid <- eval_grid[row_of_interest, ]
  
  get_event_info <- function(eval_grid) {
    load(eval_grid$file_name)
    Qmax <- dtest$Qmax[id_of_interested_event]
    event_ids <-
      cv_folds$splits[[eval_grid$iter]] %>% testing() %>% .$EventID
    event_id <- event_ids[id_of_interested_event]
    region <- eval_grid$region
    season <- eval_grid$season
    
    list(
      event_id = event_id,
      region = region,
      season = season,
      Qmax = Qmax
    )
  }
  
  
  for (i in 1:nrow(eval_grid)) {
    # load models and data
    load(eval_grid$file_name[i])
    xgb_model <- xgboost::xgb.load(eval_grid$xgb_file_name[i])
    
    names(models) <-
      plyr::mapvalues(names(models), old_model_names, new_model_names)
    
    # construct test cases
    test_cases_te <-
      lapply((49:151) * 0.01,
             prep_data_mutation,
             df = dtest,
             col_names = col_names) %>%
      lapply(function(x)
        x[id_of_interested_event, ])
    
    
    # predictions over ML models except XGBoost
    preds_tes <- tibble(model = names(models),
                        preds = vector("list", 1))
    
    for (j in seq_along(models)) {
      model <- models[[j]]
      
      preds_tes$preds[[j]] <- lapply(test_cases_te, function(x)
        predict(model, x)) %>%
        unlist()
    }
    
    # predictions of XGBoost
    preds_tes_xgb <- tibble(model = "XGBoost",
                            preds = vector("list", 1))
    
    tibble_2_DMatrix <- function(x) {
      xgb.DMatrix(data = data.matrix(x %>% select(-Qmax)),
                  label = x$Qmax)
    }
    
    preds_tes_xgb$preds[[i]] <-
      lapply(test_cases_te, function(x)
        predict(xgb_model, x %>% tibble_2_DMatrix)) %>%
      unlist()
  }
  
  
  # Background --------------------------------------------------------------
  
  event_info <- get_event_info(eval_grid)
  
  load("./mt_results/rain_mt.Rda")
  
  data_gof <- eval_grid %>%
    select(region, season, iter, gof_result) %>%
    unnest(gof_result)
  
  model_order <- data_gof %>%
    group_by(model) %>%
    dplyr::summarise(mean_gof = mean(r2)) %>%
    arrange(desc(mean_gof)) %>%
    pull(model)
  
  data_plot <- preds_tes %>%
    bind_rows(preds_tes_xgb) %>%
    unnest(preds) %>%
    mutate(x = rep(c(49:151), length(models) + 1),
           model = factor(model, levels = model_order))
  
  build_backgroud_df <- function(base_prediction_y, model) {
    base_prediction_x <- 100
    
    inconsistent_1 <- data.frame(
      x = c(base_prediction_x, base_prediction_x, Inf, Inf),
      y = c(base_prediction_y, 0, 0, base_prediction_y),
      model = rep(model, 4)
    )
    
    inconsistent_2 <-
      data.frame(
        x = c(base_prediction_x, base_prediction_x,-Inf,-Inf),
        y = c(base_prediction_y, Inf, Inf, base_prediction_y),
        model = rep(model, 4)
      )
    
    consistent_1 <-
      data.frame(
        x = c(base_prediction_x, base_prediction_x, Inf, Inf),
        y = c(base_prediction_y, Inf, Inf, base_prediction_y),
        model = rep(model, 4)
      )
    
    consistent_2 <-
      data.frame(
        x = c(base_prediction_x, base_prediction_x,-Inf,-Inf),
        y = c(base_prediction_y, 0, 0, base_prediction_y),
        model = rep(model, 4)
      )
    
    inconclusive <- data.frame(
      x = c(Inf, Inf,-Inf,-Inf),
      y = c(0,-Inf,-Inf, 0),
      model = rep(model, 4)
    )
    
    invalid <- data.frame(
      x = c(-Inf,-Inf, Inf, Inf),
      y = c(-Inf, Inf, Inf,-Inf),
      model = rep(model, 4)
    )
    
    out1 <- list(
      inconsistent_1 = inconsistent_1,
      inconsistent_2 = inconsistent_2,
      consistent_1 = consistent_1,
      consistent_2 = consistent_2,
      inconclusive = inconclusive
    )
    
    out2 <- list(invalid = invalid)
    
    if (base_prediction_y >= 0) {
      out1
    } else {
      out2
    }
  }
  
  backgroud_dfs <-
    vector("list", levels(data_plot$model) %>% length())
  for (i in 1:length(backgroud_dfs)) {
    model_name <- levels(data_plot$model)[i]
    base_prediction_y <- data_plot %>%
      dplyr::filter(model == model_name) %>%
      dplyr::slice(52) %>%
      dplyr::select(preds) %>%
      as.numeric()
    
    backgroud_dfs[[i]] <-
      build_backgroud_df(base_prediction_y, model_name)
  }
  
  inconsistent_1 <- backgroud_dfs %>%
    lapply(function(x)
      x$inconsistent_1) %>%
    bind_rows() %>%
    mutate(outcome = "Inconsistent")
  inconsistent_2 <- backgroud_dfs %>%
    lapply(function(x)
      x$inconsistent_2) %>%
    bind_rows() %>%
    mutate(outcome = "Inconsistent")
  consistent_1 <- backgroud_dfs %>%
    lapply(function(x)
      x$consistent_1) %>%
    bind_rows() %>%
    mutate(outcome = "Consistent")
  consistent_2 <- backgroud_dfs %>%
    lapply(function(x)
      x$consistent_2) %>%
    bind_rows() %>%
    mutate(outcome = "Consistent")
  inconclusive <- backgroud_dfs %>%
    lapply(function(x)
      x$inconclusive) %>%
    bind_rows() %>%
    mutate(outcome = "Inconclusive test")
  invalid <- backgroud_dfs %>%
    lapply(function(x)
      x$invalid) %>%
    bind_rows() %>%
    mutate(outcome = "Invalid")
  
  backgroud_df <- inconsistent_1 %>%
    bind_rows(inconsistent_2) %>%
    bind_rows(consistent_1) %>%
    bind_rows(consistent_2) %>%
    bind_rows(inconclusive) %>%
    bind_rows(invalid) %>%
    mutate(
      model = factor(
        model,
        levels = model_order,
        labels = replace(model_order, model_order == "CART", "CARTBag")
      ),
      outcome = factor(
        outcome,
        levels = c("Invalid", "Inconclusive test", "Inconsistent", "Consistent")
      )
    )
  
  observed_df <- tibble(
    model = factor(
      model_order,
      levels = model_order,
      labels = replace(model_order, model_order == "CART", "CARTBag")
    ),
    x = 100,
    y = event_info$Qmax,
    dummy = "Observed value"
  )
  
  data_plot <- data_plot %>%
    mutate(model = factor(
      model,
      levels = model_order,
      labels = replace(model_order, model_order == "CART", "CARTBag")
    ))
  
  n_outcome_presented <-
    backgroud_df$outcome %>% unique() %>% length()
  if (n_outcome_presented == 4) {
    p <- ggplot() +
      geom_polygon(data = backgroud_df,
                   aes(x, y, fill = outcome), alpha = 0.5) +
      geom_point(
        data = data_plot,
        aes(x, preds, group = model),
        size = 1,
        shape = 1,
        color = "grey10"
      ) +
      geom_point(
        data = observed_df,
        aes(x, y, color = dummy),
        shape = 4,
        size = 2
      ) +
      facet_wrap( ~ model) +
      scale_fill_manual(values = c("#ffbcb2",
                                   "#d0d0d0",
                                   "#ffc374",
                                   "#a3d977")) +
      labs(
        x = "Magnitude of precipitation compared to the precipitation of the orignal input sample [%]",
        y = expression(Predicted ~ italic(Q[peak]) ~ "[mm]"),
        fill = "Assessment outcome",
        color = ""
      ) +
      guides(fill = guide_legend(order = 1), col = guide_legend(order = 2)) +
      theme_bw(base_size = 10) +
      theme(legend.position = "top")
  } else {
    p <- ggplot() +
      geom_polygon(data = backgroud_df,
                   aes(x, y, fill = outcome), alpha = 0.5) +
      geom_point(
        data = data_plot,
        aes(x, preds, group = model),
        size = 1,
        shape = 1,
        color = "grey10"
      ) +
      geom_point(
        data = observed_df,
        aes(x, y, color = dummy),
        shape = 4,
        size = 2
      ) +
      facet_wrap( ~ model) +
      scale_fill_manual(values = c("#d0d0d0",
                                   "#ffc374",
                                   "#a3d977")) +
      labs(
        x = "Magnitude of precipitation compared to the precipitation of the orignal input sample [%]",
        y = expression(Predicted ~ italic(Q[peak]) ~ "[mm]"),
        fill = "Assessment outcome",
        color = ""
      ) +
      guides(fill = guide_legend(order = 1), col = guide_legend(order = 2)) +
      theme_bw(base_size = 10) +
      theme(legend.position = "top")
  }
  
  fname <- paste0("./mt_results/example_illustration_", ii, ".png")
  ggsave(
    p,
    filename = fname,
    width = 7,
    height = 5.5,
    units = "in",
    dpi = 600
  )
  
  event_info
}

set.seed(1234)

n_exp <- 100
event_info <- vector("list", n_exp)
row_of_interests <- vector("list", n_exp)
id_of_interested_events <- vector("list", n_exp)

for (i in 1:n_exp) {
  row_of_interest <-  sample(c(1:40), 1)
  id_of_interested_event <- sample(c(1:80), 1)
  
  row_of_interests[[i]] <- row_of_interest
  id_of_interested_events[[i]] <- id_of_interested_event
  
  event_info[[i]] <-
    random_figure(i, row_of_interest, id_of_interested_event)
  
  cat(i)
}

save(event_info, row_of_interests, id_of_interested_events, file = "./mt_results/random_figure_config.Rda")

"#ffbcb2"
"#d0d0d0"
"#ffc374"
"#a3d977"

# 2, 16, 24, 34, 81

# 8