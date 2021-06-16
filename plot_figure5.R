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
    "RFFit"
  )

new_model_names <-
  c(
    "LM",
    "MARSBag",
    "PLS",
    "Ridge",
    "Lasso",
    "CARTBag",
    "KNN",
    "Cubist",
    "SVMPoly",
    "SVMRadial",
    "RF"
  )


# get the order of the model in terms of prediction accuracy
load("./mt_results/mr1_mt.Rda")

data_gof <- eval_grid %>%
  select(region, season, iter, gof_result) %>%
  unnest(gof_result)

model_order <- data_gof %>%
  group_by(model) %>%
  dplyr::summarise(mean_gof = mean(r2)) %>%
  arrange(desc(mean_gof)) %>%
  pull(model)

# Function ----------------------------------------------------------------

prep_data_mutation <- function(df,
                               col_names,
                               change_ratios = 1,
                               change_amounts = 0) {
  # Change the value of "col_names" variables of "df" such that df[col_names] = df[col_names]*change_ratios + change_amounts
  # "change_ratios" and "change_amounts" can be a single number or a vector.
  # In cases they are single number, expand them to have same length as "col_names".
  
  change_ratios <- change_ratios * rep(1, length(col_names))
  change_amounts <- change_amounts * rep(1, length(col_names))
  
  for (i in seq_along(col_names)) {
    df[col_names[i]] <-
      df[col_names[i]] * change_ratios[i] + change_amounts[i]
  }
  
  df
}

get_event_info <- function(eval_grid_sub, id_of_interested_event, cv_folds, dtest) {
  # Get the basic information of the interested flood event
  
  # eval_grid_sub: table that stores the modeling results, it only have 1 row (i.e., the interested outer CV)
  # id_of_interested_event: the row number of "dtest", which is the test set
  # cv_folds: "rsample" object that stores the resampling info
  # dtest: test set
  
  # extract result
  Qmax <- dtest$Qmax[id_of_interested_event] # observed Qmax
  event_ids <- cv_folds$splits[[eval_grid_sub$iter]] %>% testing() %>% .$EventID
  event_id <- event_ids[id_of_interested_event] # event ID
  region <- eval_grid_sub$region 
  season <- eval_grid_sub$season
  
  # return
  list(
    event_id = event_id,
    region = region,
    season = season,
    Qmax = Qmax
  )
}

random_figure <- function(ii,
                          row_of_interest = 17,
                          id_of_interested_event = 19) {
  # Plot changes in Qmax predictions under changing precipitation
  # ii: suffix of file name of the figure
  # row_of_interest: the row number of "eval_grid", which store the results of different CV iterations
  # id_of_interested_event: the row number of "dtest", which is the test set.

  #   eval_grid: Tibble of region, season, iteration number, name of the file storing the model 
  # raining example, and XGBoost model.
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
    )
  
  # names of the variables whose values are to be changes. They correspond to column names of "dtest"
  col_names <- c("P0", "P1", "P3", "P5", "P7", "Pmin", "Pmax")
  
  # experiments
  # subset "eval_grid" to keep the interested row only
  eval_grid <- eval_grid[row_of_interest,]
  
  # load models and data
  load(eval_grid$file_name)
  
  # rename model name for plotting: drop "Fit" suffix and other improvement
  names(models) <-
    plyr::mapvalues(names(models), old_model_names, new_model_names)
  
  # construct inputs from the training and the test set for assessment.
  # "test_cases_te": inputs derived from the events of the test set.
  # Variables of "col_names" are changed to between 49% and 151% of the original values;
  # and the inputs correspond to "id_of_interested_event" are kept.
  # The updated inputs are stored in a list format.
  
  test_cases_te <-
    lapply((49:151) * 0.01,
           prep_data_mutation,
           df = dtest,
           col_names = col_names) %>%
    lapply(function(x)
      x[id_of_interested_event,])
  
  
  # Predictions associated with "test_cases_te" are stored in "preds_te".
  # The result of each model expect XGBoost (which is processed later) is store as an element of a list.
  preds_tes <- tibble(model = names(models),
                      preds = vector("list", 1))
  
  for (j in seq_along(models)) {
    # iterate over the models
    model <- models[[j]]
    
    # use apply to iterates over the modified inputs, and store it in preds_tes
    preds_tes$preds[[j]] <- lapply(test_cases_te, function(x)
      predict(model, x)) %>%
      unlist()
  }
  
  # predictions of XGBoost
  # load model
  xgb_model <- xgb.load(eval_grid$xgb_file_name)
  
  # get the basic info of the event
  event_info <- get_event_info(eval_grid, id_of_interested_event, cv_folds, dtest)
  
  # "preds_tes_xgb" stores the prediction results
  preds_tes_xgb <- tibble(model = "XGBoost",
                          preds = vector("list", 1))
  
  # convert to xgb.DMatrix format for xgboost
  tibble_2_DMatrix <- function(x) {
    xgb.DMatrix(data = data.matrix(x %>% select(-Qmax)),
                label = x$Qmax)
  }
  # save the results to a list 
  preds_tes_xgb$preds[[1]] <-
    lapply(test_cases_te, function(x)
      predict(xgb_model, x %>% tibble_2_DMatrix)) %>%
    unlist()
  
  # join the results of different models, and add time steps
  data_plot <- preds_tes %>%
    bind_rows(preds_tes_xgb) %>%
    unnest(preds) %>%
    mutate(x = rep(c(49:151), length(models) + 1),
           model = factor(model, levels = model_order))
  
  # prepare the colored background and other required info for plotting
  build_backgroud_df <- function(base_prediction_y, model) {
    # base_prediction_y is the prediction of the original input
    
    base_prediction_x <-
      100 # the prediction of 100% of the original pp is interested
    
    # background for inconsistent predictions
    inconsistent_1 <- data.frame(
      x = c(base_prediction_x, base_prediction_x, Inf, Inf),
      y = c(base_prediction_y, 0, 0, base_prediction_y),
      model = rep(model, 4)
    )
    
    inconsistent_2 <-
      data.frame(
        x = c(base_prediction_x, base_prediction_x, -Inf, -Inf),
        y = c(base_prediction_y, Inf, Inf, base_prediction_y),
        model = rep(model, 4)
      )
    
    # background for consistent predictions
    consistent_1 <-
      data.frame(
        x = c(base_prediction_x, base_prediction_x, Inf, Inf),
        y = c(base_prediction_y, Inf, Inf, base_prediction_y),
        model = rep(model, 4)
      )
    
    consistent_2 <-
      data.frame(
        x = c(base_prediction_x, base_prediction_x, -Inf, -Inf),
        y = c(base_prediction_y, 0, 0, base_prediction_y),
        model = rep(model, 4)
      )
    
    # background for inconclusive test predictions
    inconclusive <- data.frame(
      x = c(Inf, Inf, -Inf, -Inf),
      y = c(0, -Inf, -Inf, 0),
      model = rep(model, 4)
    )
    
    # background for invlid test predictions
    invalid <- data.frame(
      x = c(-Inf, -Inf, Inf, Inf),
      y = c(-Inf, Inf, Inf, -Inf),
      model = rep(model, 4)
    )
    
    # if the prediction of the interested event is valid, i.e., >=0
    out1 <- list(
      inconsistent_1 = inconsistent_1,
      inconsistent_2 = inconsistent_2,
      consistent_1 = consistent_1,
      consistent_2 = consistent_2,
      inconclusive = inconclusive
    )
    
    # if the prediction of the interested event is invalid, i.e., < 0
    out2 <- list(invalid = invalid)
    
    if (base_prediction_y >= 0) {
      out1
    } else {
      out2
    }
  }
  
  # prepare background color dataframe for each model
  backgroud_dfs <- vector("list", length(levels(data_plot$model)))
  for (i in 1:length(backgroud_dfs)) {
    model_name <- levels(data_plot$model)[i]
    base_prediction_y <- data_plot %>%
      dplyr::filter(model == model_name) %>%
      dplyr::slice(52) %>% # the prediction for the original event 
      dplyr::select(preds) %>%
      as.numeric()
    
    backgroud_dfs[[i]] <-
      build_backgroud_df(base_prediction_y, model_name)
  }
  
  # join the background color dataframe of all models
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
    mutate(outcome = "Invalid") # "invalid" can be empty
  
  backgroud_df <- inconsistent_1 %>%
    bind_rows(inconsistent_2) %>%
    bind_rows(consistent_1) %>%
    bind_rows(consistent_2) %>%
    bind_rows(inconclusive) %>%
    bind_rows(invalid) %>%
    mutate(
      model = factor(model,
                     levels = model_order),
      outcome = factor(
        outcome,
        levels = c("Invalid", "Inconclusive test", "Inconsistent", "Consistent")
      )
    )
  
# dataframe of the observed Qmax
  observed_df <- tibble(
    model = factor(model_order,
                   levels = model_order),
    x = 100,
    y = event_info$Qmax,
    dummy = "Observed value"
  )
  
  # "n_outcome_presented" count the number of possible outcome presented in the plot
  # n_outcome_presented==4 means "invalid" outcome is presented
  # n_outcome_presented==4 means "invalid" outcome is not presented
  # This information is useful for coloring the background
  
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
      facet_wrap(~ model) +
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
      facet_wrap(~ model) +
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
  
  fname <- paste0("./paper_figures/example_illustration_", ii, ".jpg")
  ggsave(
    p,
    filename = fname,
    width = 7,
    height = 5.5,
    units = "in",
    dpi = 300
  )
  
  event_info
}


# Plotting ----------------------------------------------------------------


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
  
  cat(i, "\n")
}

# save the information on the random events selected for plotting
save(event_info, row_of_interests, id_of_interested_events, file = "./mt_results/random_figure_config.Rda")

# The colors of the background
"#ffbcb2"
"#d0d0d0"
"#ffc374"
"#a3d977"

# Figure number of the figure presented in the main text of the paper 
# 8

# Figure numbers of the figures presented in the supplement of the paper 
# 2, 16, 24, 34, 81