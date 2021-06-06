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
  hydroGOF
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


# Data --------------------------------------------------------------------

#   eval_grid: Tibble of region, season, iteration number, name of the file storing the model 
# raining example, and XGBoost model. "gof_result" stores the goodness of fit results on the test datasets.
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
  # load the file store the model training results. "models" stores the model training results.
  load(eval_grid$file_name[i])
  
  # "df1" stores goodness-of-fit (gof) measured by R2 of all models except XGBoost
  df1 <-
    sapply(models, function(x)
      hydroGOF::gof(predict(x, dtest), dtest$Qmax, digits = 8)[17]) %>%
    as_tibble() %>%
    mutate(model = names(models))
  
  # gof of XGBoost
  # Convert "dtest" to xgb.DMatrix format
  xgb_dtest <-
    xgb.DMatrix(data = data.matrix(dtest %>% select(-Qmax)),
                label = dtest$Qmax)
  
  # load XGBoost
  xgb_model <- xgboost::xgb.load(eval_grid$xgb_file_name[i])
  
  # "df2" stores goodness-of-fit (gof) measured by R2 of XGBoost
  df2 <- tibble(model = "XGBoost",
                value = hydroGOF::gof(
                  sim = predict(xgb_model, xgb_dtest),
                  ob = dtest$Qmax,
                  digits = 8
                )[17])
  
  # Combine "df1" and "df2" and put it into "eval_grid"; Change model name to new names
  eval_grid$gof_result[[i]] <- bind_rows(df1, df2) %>%
    dplyr::rename(r2 = value) %>%
    dplyr::select(model, r2) %>%
    mutate(model = plyr::mapvalues(model, old_model_names, new_model_names))
}


# Rain MT -----------------------------------------------------------------

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

rain_mt <- function(x, x_low, x_high) {
  # Function to conduct MT based assessment.
  # "x" is the vector of predictions for the event of interest.
  # "x_low" is the vector of predictions for the event with a lower precipitation magnitude.
  # "x_high" is the vector of predictions for the event with a higher precipitation magnitude.
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

# column names of the variables whose values are to be changed.
col_names <- c("P0", "P1", "P3", "P5", "P7", "Pmin", "Pmax")

# add columns to "eval_grid" to store the results for the training and the test set.
eval_grid <- eval_grid %>%
  mutate(mt_trs = vector("list", 1),
         mt_tes = vector("list", 1))

# iteration over different region, season, and outer CV
for (i in 1:nrow(eval_grid)) {
  # load models and data
  load(eval_grid$file_name[i])
  xgb_model <- xgboost::xgb.load(eval_grid$xgb_file_name[i])
  
  # rename models
  names(models) <-
    plyr::mapvalues(names(models), old_model_names, new_model_names)
  
  # construct test cases
  # "test_cases_te": inputs derived from the events of the test set.
  # "test_cases_tr": inputs derived from the events of the training set.
  # Variables of "col_names" are changed to between 49% and 151% of the original values.
  # The updated inputs are stored in a list format.
  test_cases_te <-
    lapply((49:151) * 0.01,
           prep_data_mutation,
           df = dtest,
           col_names = col_names)
  
  test_cases_tr <-
    lapply((49:151) * 0.01,
           prep_data_mutation,
           df = dtrain,
           col_names = col_names)
  
  # assessment result for each model is stored in a list.
  mt_tes <- vector("list", length(models) + 1)
  mt_trs <- vector("list", length(models) + 1)
  
  names(mt_tes) <- c(names(models), "XGBoost")
  names(mt_trs) <- c(names(models), "XGBoost")
  
  # Iterate over ML models except XGBoost
  for (j in seq_along(models)) {
    model <- models[[j]]
    
    # Predictions associated with the test set and the training set are stored in "preds_te" and "preds_tr".
    # The result of each precipitation change rate is store as an element of a list.
    preds_te <- lapply(test_cases_te, function(x)
      predict(model, x))
    preds_tr <- lapply(test_cases_tr, function(x)
      predict(model, x))
    
    # "mt_te" stores the assessment results associated with the test set.
    # "mt_tr" stores the assessment results associated with the training set.
    mt_te <-
      vector("list", length(preds_te) - 2) # the first and the last test case can't be source input
    mt_tr <-       
      vector("list", length(preds_tr) - 2) # the first and the last test case can't be source input
    
    # iteration over different test cases
    for (k in 2:(length(preds_te) - 1)) {
      # multiple assignment, 3 elements of the predictions are selected.
      c(x_low, x, x_high) %<-% preds_te[(k - 1):(k + 1)]
      mt_te[[k - 1]] <- rain_mt(x, x_low, x_high)
      
      c(x_low, x, x_high) %<-% preds_tr[(k - 1):(k + 1)]
      mt_tr[[k - 1]] <- rain_mt(x, x_low, x_high)
    }
    
    # Store the result for each ML model.
    mt_tes[[j]] <- mt_te
    mt_trs[[j]] <- mt_tr
  }
  
  # MT for XGBoost
  # function to convert tibble to xgb.DMatrix
  tibble_2_DMatrix <- function(x) {
    xgb.DMatrix(data = data.matrix(x %>% select(-Qmax)),
                label = x$Qmax)
  }
  
  # "preds_te" and "preds_tr" store all the predictions.
  preds_te <-
    lapply(test_cases_te, function(x)
      predict(xgb_model, x %>% tibble_2_DMatrix))
  preds_tr <-
    lapply(test_cases_tr, function(x)
      predict(xgb_model, x %>% tibble_2_DMatrix))
  
  # "mt_te" and "mt_tr" store all the assessment results.
  mt_te <- vector("list", length(preds_te) - 2)
  mt_tr <- vector("list", length(preds_tr) - 2)
  
  for (k in 2:(length(preds_te) - 1)) {
    c(x_low, x, x_high) %<-% preds_te[(k - 1):(k + 1)]
    mt_te[[k - 1]] <- rain_mt(x, x_low, x_high)
    
    c(x_low, x, x_high) %<-% preds_tr[(k - 1):(k + 1)]
    mt_tr[[k - 1]] <- rain_mt(x, x_low, x_high)
  }
  
  mt_tes$XGBoost <- mt_te
  mt_trs$XGBoost <- mt_tr
  
  # Save the results for all models in "eval_grid".
  eval_grid$mt_tes[[i]] <- mt_tes
  eval_grid$mt_trs[[i]] <- mt_trs
}

# Save the assessment results for further analysis.
save(eval_grid, file = "./mt_results/mr1_mt.Rda")

