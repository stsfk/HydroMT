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



# Function ----------------------------------------------------------------


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

mt_pet <- function(x, x_low, x_high) {
  # inconclusive = 1, invalid = 2, inconsistent = 3, consistent = 4
  
  out <- rep(3, length(x))
  
  # inconclusive
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
  
  # invalid
  invalid_index <-
    c(which(is.na(x)), which(is.nan(x)), which(x < 0)) %>%
    unique()
  
  # inconsistent or consistent
  consistent_index <-
    ((x <= x_low) & (x >= x_high)) %>%
    which() %>%
    unique()
  
  # Mark consistent
  out[consistent_index] <- 4
  
  # Mark inconclusive
  out[inconclusive_index] <- 1
  
  # Mark invalid
  out[invalid_index] <- 2
  
  # return
  out
}

mt_p <- function(x, x_low, x_high) {
  # inconclusive = 1, invalid = 2, inconsistent = 3, consistent = 4
  
  
  out <- rep(3, length(x))
  
  # inconclusive
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
  
  # invalid
  invalid_index <-
    c(which(is.na(x)), which(is.nan(x)), which(x < 0)) %>%
    unique()
  
  # inconsistent or consistent
  consistent_index <-
    ((x >= x_low) & (x <= x_high)) %>%
    which() %>%
    unique()
  
  # Mark consistent
  out[consistent_index] <- 4
  
  # Mark inconclusive
  out[inconclusive_index] <- 1
  
  # Mark invalid
  out[invalid_index] <- 2
  
  # return
  out
}


# MT ----------------------------------------------------------------------

col_names_p <- c("P0", "P1", "P3", "P5", "P7", "Pmin", "Pmax")
col_names_pet <- c("PET0", "PET1", "PET3", "PET5", "PET7")

eval_grid <- eval_grid %>%
  mutate(mt_tes = vector("list", 1))

# iteration over different region, season, and outer CV
for (i in 1:nrow(eval_grid)) {
  # load models and data
  load(eval_grid$file_name[i])
  xgb_model <- xgboost::xgb.load(eval_grid$xgb_file_name[i])
  
  names(models) <-
    plyr::mapvalues(names(models), old_model_names, new_model_names)
  
  # construct test cases, modify p cols first, then PET cols
  test_cases_te <-
    lapply((49:151) * 0.01,
           prep_data_mutation,
           df = dtest,
           col_names = col_names_p)
  
  
  test_cases_te_pet_plus <- test_cases_te %>%
    lapply(function(x)
      prep_data_mutation(
        df = x,
        col_names = col_names_pet,
        change_ratios = 1.01
      ))
  
  
  test_cases_te_pet_minus <- test_cases_te %>%
    lapply(function(x)
      prep_data_mutation(
        df = x,
        col_names = col_names_pet,
        change_ratios = 0.99
      ))
  
  
  # mt result for each model
  mt_tes <- vector("list", length(models) + 1)

  names(mt_tes) <- c(names(models), "XGBoost")

  # iteration over ML models except XGBoost
  for (j in seq_along(models)) {
    model <- models[[j]]
    
    preds_te <- lapply(test_cases_te, function(x)
      predict(model, x))
    
    preds_plus <- lapply(test_cases_te_pet_plus, function(x)
      predict(model, x))
    
    preds_minus <- lapply(test_cases_te_pet_minus, function(x)
      predict(model, x))
    
    mt_te <- vector("list", length(test_cases_te) - 2) # the first and the last test case can't be source input
    # iteration over different test cases
    for (k in 2:(length(preds_te) - 1)) {
      
      c(x_low, x, x_high) %<-% list(preds_minus[[k]], preds_te[[k]], preds_plus[[k]])
      mt_te[[k - 1]] <- mt_pet(x, x_low, x_high)
    }
    
    mt_tes[[j]] <- mt_te
  }
  
  # MT for XGBoost
  tibble_2_DMatrix <- function(x) {
    xgb.DMatrix(data = data.matrix(x %>% select(-Qmax)),
                label = x$Qmax)
  }
  
  preds_te <-
    lapply(test_cases_te, function(x)
      predict(xgb_model, x %>% tibble_2_DMatrix))
  
  preds_plus <-
    lapply(test_cases_te_pet_plus, function(x)
      predict(xgb_model, x %>% tibble_2_DMatrix))
  
  preds_minus <-
    lapply(test_cases_te_pet_minus, function(x)
      predict(xgb_model, x %>% tibble_2_DMatrix))
  
  mt_te <- vector("list", length(test_cases_te) - 2)

  for (k in 2:(length(preds_te) - 1)) {
    c(x_low, x, x_high) %<-% list(preds_minus[[k]], preds_te[[k]], preds_plus[[k]])
    mt_te[[k - 1]] <- mt_pet(x, x_low, x_high)
  }
  
  mt_tes$XGBoost <- mt_te

  # output
  eval_grid$mt_tes[[i]] <- mt_tes
}

save(eval_grid, file = "./mt_results/p_pet_mt.Rda")

