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


# MR3 - relation between predicted values for different samples --------

# add columns to "eval_grid" to store the results for the test set.
eval_grid <- eval_grid %>%
  mutate(mt_tes = vector("list", 1))


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
  # consistent rate of predictions of all models except xgboost
  load(eval_grid$file_name[i])
  df1 <- sapply(models, compute_consist_rate, dtest, xgboost_format = F)
  
  # consistent rate of predictions of xgboost model
  xgb_model <- xgboost::xgb.load(eval_grid$xgb_file_name[i])
  df2 <- compute_consist_rate(xgb_model, dtest, xgboost_format=T)
  
  # assign result
  eval_grid$mt_tes[[i]] <- tibble(
    model = c(names(df1), "XGBoost"),
    consistent_rate = c(df1, df2))%>%
    mutate(model = plyr::mapvalues(model, old_model_names, new_model_names))
}

save(eval_grid, file = "./mt_results/mr3_mt.Rda")


# Ratio of observations with non-unique observations ----------------------
uniqe_ratios <- rep(0, nrow(eval_grid))
for (i in 1:nrow(eval_grid)) {
  # consistent rate of predictions of all models except xgboost
  load(eval_grid$file_name[i])
  
  qmax_table <- table(dtest$Qmax) %>% unname()
  
  uniqe_ratios[[i]] <- sum(qmax_table == 1)/length(qmax_table)
}

max(1 - uniqe_ratios) # which is at most 5% of the observations have non-unique values
# assume the 5% observations all have the same prediction 
# In region 3 winter flood model iteration 5, among the 743 samples of the test set, 32 have non-unique values
# Assume all the 32 non-unique observations are the same
# they correspond to C(32,2) = 496 assessments
# and the total number of possible assessments is C(743, 2) = 275653
496/275653 = 0.001799364
# That is, relaxing the requirement for events with non-unique value prediction 
# can only affects a few small portion of the assessments.





