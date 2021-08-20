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
  lemon
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
      "./modeling_results/mono/xgb_region",
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
  xgb_model <- xgboost::xgb.load(eval_grid$xgb_file_name[i])
  
  # gof of xgboost model
  xgb_dtest <-
    xgb.DMatrix(data = data.matrix(dtest %>% select(-Qmax)),
                label = dtest$Qmax)
  

  eval_grid$gof_result[[i]]  <- tibble(model = "XGBoost",
                value = hydroGOF::gof(
                  sim = predict(xgb_model, xgb_dtest),
                  ob = dtest$Qmax,
                  digits = 8
                )[17]) %>%
    dplyr::rename(r2 = value) %>%
    dplyr::select(model, r2)
}


data_r2 <- eval_grid %>%
  select(region, season, iter, gof_result) %>%
  unnest(gof_result) %>%
  dplyr::mutate(model = "XGBoost_mono")


# PP MT -----------------------------------------------------------------


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

col_names <- c("P0", "P1", "P3", "P5", "P7", "Pmin", "Pmax")

pp_mt <- function(x, x_low, x_high) {
  # Function to conduct MT based assessment.
  # "x" is the vector of predictions for the event of interest.
  # "x_low" is the vector of predictions for the events with lower precipitation magnitudes.
  # "x_high" is the vector of predictions for the events with higher precipitation magnitudes.
  # Assessment outcome:
  # inconclusive test = 1, invalid = 2, inconsistent = 3, consistent = 4
  
  out <- rep(3, length(x))
  
  # inconclusive: "x_low" or "x_high" is invalid prediction.
  inconclusive_index <-
    c(
      which(is.na(x_low)),
      which(is.nan(x_low)),
      which(x_low < 0),
      which(is.na(x_high)),
      which(is.nan(x_high)),
      which(x_high < 0)
    ) %>%
    unique()
  
  # invalid: "x" is invalid prediction; The result overwrites that for "inconclusive", as it has higher priority.
  invalid_index <-
    c(which(is.na(x)), which(is.nan(x)), which(x < 0)) %>%
    unique()
  
  # inconsistent or consistent: consistent_index is computed; 
  # "(x >= x_low) & (x <= x_high)" is required by domain knowledge.
  consistent_index <-
    ((x >= x_low) & (x <= x_high)) %>%
    which() %>%
    unique()
  
  # Mark consistent
  out[consistent_index] <- 4
  
  # Mark inconclusive test; overwrites consistent
  out[inconclusive_index] <- 1
  
  # Mark invalid; overwrites inconclusive test and consistent
  out[invalid_index] <- 2
  
  # return
  out
}

eval_grid <- eval_grid %>%
  mutate(mt_tes = vector("list", 1))

# iteration over different region, season, and outer CV
for (i in 1:nrow(eval_grid)) {
  # load models and data
  load(eval_grid$file_name[i])
  xgb_model <- xgboost::xgb.load(eval_grid$xgb_file_name[i])
  
  # construct test cases
  test_cases_te <-
    lapply((49:151) * 0.01,
           prep_data_mutation,
           df = dtest,
           col_names = col_names)
  
  # MT for XGBoost
  tibble_2_DMatrix <- function(x) {
    xgb.DMatrix(data = data.matrix(x %>% select(-Qmax)),
                label = x$Qmax)
  }
  
  preds_te <-
    lapply(test_cases_te, function(x)
      predict(xgb_model, x %>% tibble_2_DMatrix))
  
  mt_te <- vector("list", length(test_cases_te) - 2)

  for (k in 2:(length(preds_te) - 1)) {
    c(x_low, x, x_high) %<-% preds_te[(k - 1):(k + 1)]
    mt_te[[k - 1]] <- pp_mt(x, x_low, x_high)
  }

  # output
  eval_grid$mt_tes[[i]] <- mt_te
}

eval_grid <- eval_grid %>%
  mutate(mr1 = 0)

get_consistency_rate <- function(x) {
  (sum(x == 4)) / length(x)
}

for (i in 1:nrow(eval_grid)){
  
  eval_grid$mr1[i] <- eval_grid$mt_tes[[i]] %>%
    unlist() %>%
    get_consistency_rate()
}

data_mr1 <- eval_grid %>%
  select(region, season, iter, mr1) %>%
  dplyr::mutate(model = "XGBoost_mono")


# PET MT ------------------------------------------------------------------

col_names <- c("PET0", "PET1", "PET3", "PET5", "PET7")

eval_grid <- eval_grid %>%
  mutate(mt_tes = vector("list", 1))

pet_mt <- function(x, x_low, x_high) {
  # Function to conduct MT based assessment.
  # "x" is the vector of predictions for the event of interest.
  # "x_low" is the vector of predictions for the events with lower PET magnitudes.
  # "x_high" is the vector of predictions for the events with higher PET magnitudes.
  # Assessment outcome:
  # inconclusive test = 1, invalid = 2, inconsistent = 3, consistent = 4
  
  out <- rep(3, length(x))
  
  # inconclusive: "x_low" or "x_high" is invalid prediction.
  inconclusive_index <-
    c(
      which(is.na(x_low)),
      which(is.nan(x_low)),
      which(x_low < 0),
      which(is.na(x_high)),
      which(is.nan(x_high)),
      which(x_high < 0)
    ) %>%
    unique()
  
  # invalid: "x" is invalid prediction; The result overwrites that for "inconclusive", as it has higher priority.
  invalid_index <-
    c(which(is.na(x)), which(is.nan(x)), which(x < 0)) %>%
    unique()
  
  # inconsistent or consistent: consistent_index is computed; 
  # "(x <= x_low) & (x >= x_high)" is required by domain knowledge.
  consistent_index <-
    ((x <= x_low) & (x >= x_high)) %>%
    which() %>%
    unique()
  
  # Mark consistent
  out[consistent_index] <- 4
  
  # Mark inconclusive test; overwrites consistent
  out[inconclusive_index] <- 1
  
  # Mark invalid; overwrites inconclusive test and consistent
  out[invalid_index] <- 2
  
  # return
  out
}

# iteration over different region, season, and outer CV
for (i in 1:nrow(eval_grid)) {
  # load models and data
  load(eval_grid$file_name[i])
  xgb_model <- xgboost::xgb.load(eval_grid$xgb_file_name[i])
  
  # construct test cases
  test_cases_te <-
    lapply((49:151) * 0.01,
           prep_data_mutation,
           df = dtest,
           col_names = col_names)
  
  # MT for XGBoost
  tibble_2_DMatrix <- function(x) {
    xgb.DMatrix(data = data.matrix(x %>% select(-Qmax)),
                label = x$Qmax)
  }
  
  preds_te <-
    lapply(test_cases_te, function(x)
      predict(xgb_model, x %>% tibble_2_DMatrix))

  mt_te <- vector("list", length(test_cases_te) - 2)

  for (k in 2:(length(preds_te) - 1)) {
    c(x_low, x, x_high) %<-% preds_te[(k - 1):(k + 1)]
    mt_te[[k - 1]] <- pet_mt(x, x_low, x_high)
  }
  
  # output
  eval_grid$mt_tes[[i]] <- mt_te
}

eval_grid <- eval_grid %>%
  mutate(mr2 = 0)

get_consistency_rate <- function(x) {
  (sum(x == 4)) / length(x)
}

for (i in 1:nrow(eval_grid)){
  eval_grid$mr2[i] <- eval_grid$mt_tes[[i]] %>%
    unlist() %>%
    get_consistency_rate()
}

data_mr2 <- eval_grid %>%
  select(region, season, iter, mr2) %>%
  dplyr::mutate(model = "XGBoost_mono")


# MR3 ---------------------------------------------------------------------

eval_grid <- eval_grid %>%
  mutate(mr3 = 0)

compute_consist_rate <- function(model, dtest, xgboost_format = F){
  # Input: model, dtest: test set, xgboost_format:whether the test set is coverted into xgb.DMatrix
  # Output: consistent rate, i.e., 
  
  # number of tests for each sample, i.e., number of samples - 1
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
  
  # get only valid predictions and the corresponding observations, this is to speed-up the computation
  # so the number of "invalid" and "inconclusive test" cases do not need to be counted 
  valid_index <- (prediction >= 0)
  observation <- observation[valid_index]
  prediction <- prediction[valid_index] 
  
  # compute the number of consistent predictions for each sample
  # the number of consistent outcome when a sample is used in prediction of each sample
  n_consistents <- vector("double", length(prediction)) 
  for (i in 1:length(prediction)){
    # compute n_inconsistent first; 
    # (prediction[i] - prediction) is the difference between the prediction of the event of interest -- prediction[i] 
    #  and that of the other samples;
    # (observation[i] - observation) is the difference between the observation of the event of interest --observation[i] 
    # and that of the other samples;
    # if (prediction[i] - prediction[j])*(observation[i] - observation[j]) < 0, then prediction[i] is inconsistent when
    # prediction[j] is used as the follow-up input;
    # in rare cases that observation[i] = observation[j], any valid prediction is considered consistent 
    n_inconsisitent <- ((prediction[i] - prediction)*(observation[i] - observation) < 0) %>% sum() 
    n_consistents[i] <- length(prediction) - n_inconsisitent - 1 # total - inconsistent - self
  }
  
  # output, the average consistent rate of valid predictions 
  # "/n_tests" is the number of all possible assessment for each sample
  out <- (n_consistents/n_tests) %>%
    mean()
  
  # the average consistent rate computed for all samples
  out*length(prediction)/nrow(dtest)
}


for (i in 1:nrow(eval_grid)) {
  # gof of all models except xgboost
  load(eval_grid$file_name[i])
  
  # gof of xgboost model
  xgb_model <- xgboost::xgb.load(eval_grid$xgb_file_name[i])
  
  df2 <- compute_consist_rate(xgb_model, dtest, xgboost_format=T)
  
  # assign result
  eval_grid$mr3[i] <- df2
}

data_mr3 <- eval_grid %>%
  select(region, season, iter, mr3) %>%
  dplyr::mutate(model = "XGBoost_mono")


# Combine -----------------------------------------------------------------

data_mono <- data_r2 %>%
  left_join(data_mr1, by = c("region", "season", "iter", "model")) %>%
  left_join(data_mr2, by = c("region", "season", "iter", "model")) %>%
  left_join(data_mr3, by = c("region", "season", "iter", "model"))

save(data_mono, file = "./mt_results/mono_mt.Rda")
