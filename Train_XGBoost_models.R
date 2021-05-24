# Reference: skopt.gp_minimize https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html#skopt.gp_minimize

source("./Preprocessing.R")

# train models as functions

prepare_train_test_data <-
  function(outer_i,
           data_process,
           cv_folds,
           xgboost_format = F,
           return_index = T) {
    # extract folds from data_feature for training, validation, and testing
    training_outer <- analysis(cv_folds$splits[[outer_i]])
    test <- assessment(cv_folds$splits[[outer_i]])
    
    # data only training and testing are named as dtrain and dtest,
    # training_outer_eventID and test_eventID store the unique EventID for each flood event
    training_outer_eventID <- training_outer$EventID
    test_eventID <- test$EventID
    
    dtrain <- training_outer %>% select(-EventID)
    dtest <- test %>% select(-EventID)
    
    # get relative location of validation in training_outer data sets
    val <-
      lapply(cv_folds$inner_resamples[[outer_i]]$splits, analysis) # get training_inner
    val <-
      lapply(val, function(x)
        x %>% pull(EventID)) # get EventID of training_inner
    val <-
      lapply(val, function(x)
        which(training_outer_eventID %in% x)) # get relative location
    
    # convert train_outer and test to DMatrix
    if (xgboost_format) {
      val <-
        lapply(cv_folds$inner_resamples[[outer_i]]$splits, assessment)  # get validation folds
      val <-
        lapply(val, function(x)
          x %>% pull(EventID))  # get EventID of validation
      val <-
        lapply(val, function(x)
          which(training_outer_eventID %in% x))# get relative location
      
      # convert train_outer and test to DMatrix
      dtrain <- training_outer %>%
        dplyr::select(-EventID)
      
      dtest <- test %>%
        dplyr::select(-EventID)
    }
    
    # return
    if (return_index) {
      out <- list(
        dtrain = dtrain,
        dtest = dtest,
        val = val,
        train_index = training_outer_eventID,
        test_index = test_eventID
      )
    } else {
      out <- list(dtrain = dtrain,
                  dtest = dtest,
                  val = val)
    }
    
    return(out)
  }

scoringFunction <-
  function(eta,
           max_depth,
           min_child_weight,
           subsample,
           colsample_bytree,
           gamma,
           final_model = T) {
    dtrain <- xgb.DMatrix(data = data.matrix(dtrain %>% select(-Qmax)),
                          label = dtrain$Qmax)
    
    dtest <-
      xgb.DMatrix(data = data.matrix(dtest %>% select(-Qmax)),
                  label = dtest$Qmax)
    
    # xgboost hyperparameters and CV training
    Pars <- list(
      booster = "gbtree",
      eta = eta,
      max_depth = max_depth,
      min_child_weight = min_child_weight,
      colsample_bytree = colsample_bytree,
      subsample = subsample,
      gamma = gamma
    )
    
    xgbcv <- xgb.cv(
      objective = "reg:squarederror",
      data = dtrain,
      folds = val,
      tree_method = tree_method,
      max_bin = 256,
      nround = 5000,
      early_stopping_rounds = 20,
      verbose = 0,
      params = Pars
    )
    
    
    # train model on all training_outer fold
    pred_df <- list()
    if (final_model) {
      watchlist <- NULL
      xgbFit <- xgb.train(
        data = dtrain,
        objective = "reg:squarederror",
        tree_method = tree_method,
        max_bin = 256,
        nround = xgbcv$best_iteration,
        verbose = 0,
        params = Pars
      )
      
      pred_df <- tibble(ob = getinfo(dtest, "label"),
                        pred = predict(xgbFit, dtest))
    }
    
    # output
    out <- list(
      Score = -min(xgbcv$evaluation_log$test_rmse_mean)
      ,
      nrounds = xgbcv$best_iteration
      ,
      pred_df = list(pred_df)
    )
    
    gc()
    
    return(out)
  }

final_xgb_model <- function(optObj, region, season, outer_i) {
  # This function return the best model according to the Bayesian optimization results
  dtrain <-
    xgb.DMatrix(data = data.matrix(dtrain %>% select(-Qmax)),
                label = dtrain$Qmax)
  dtest <- xgb.DMatrix(data = data.matrix(dtest %>% select(-Qmax)),
                       label = dtest$Qmax)
  
  xgbFit <- xgb.train(
    data = dtrain,
    objective = "reg:squarederror",
    tree_method = tree_method,
    max_bin = 256,
    nround = optObj$scoreSummary$nrounds[which.max(optObj$scoreSummary$Score)],
    verbose = 0,
    params = getBestPars(optObj)
  )
  
  xgboost::xgb.save(
    xgbFit,
    fname = paste0(
      "./modeling_results/xgb_region",
      region,
      "_",
      season,
      "_iter",
      outer_i,
      ".model"
    )
  )
  xgboost::xgb.DMatrix.save(
    dtrain,
    fname = paste0(
      "./modeling_results/xgb_region",
      region,
      "_",
      season,
      "_iter",
      outer_i,
      "_tr.data"
    )
  )
  xgboost::xgb.DMatrix.save(
    dtest,
    fname = paste0(
      "./modeling_results/xgb_region",
      region,
      "_",
      season,
      "_iter",
      outer_i,
      "_te.data"
    )
  )
  
  gc()
  
  T
}

# Evaluation --------------------------------------------------------------

SEED = 439759
outer_repeats = 1
inner_repeats = 1
outer_v = 5
inner_v = 5

tree_method = "hist"

bounds <- list(
  eta = c(0.005, 0.1),
  max_depth = c(2L, 10L),
  min_child_weight = c(1L, 10L),
  subsample = c(0.25, 1),
  colsample_bytree = c(0.25, 1),
  gamma = c(0, 10)
)

input_df <- expand_grid(region = c(1:4),
                        season = c("S", "W"))

initPoints <- 30
max_iter <- 100
patience <- 10
plotProgress <- F

start_time <- Sys.time()
for (i in 1:nrow(input_df)) {
  # i is for each region and season
  
  region <- input_df$region[i]
  season <- input_df$season[i]
  
  # read data and preprocess
  file_names <- list(
    paste0("./raw_data/5_Reg", region, "_Tr_", season, ".csv"),
    paste0("./raw_data/5_Reg", region, "_Te_", season, ".csv")
  )
  
  c(data_process, data_raw) %<-% read_data_single_catchment(file_names)
  
  data_process <- data_process %>%
    dplyr::select(-Season,-Region)
  recipe_all <- recipe(Qmax ~ ., data = data_process) %>%
    step_dummy(all_nominal())
  recipe_all <- prep(recipe_all, training = data_process)
  data_process <- bake(recipe_all, new_data = data_process)
  data_process["EventID"] <- data_raw$EventID
  
  # split data
  set.seed(SEED)
  cv_folds <- nested_cv(
    data_process,
    outside = vfold_cv(
      v = outer_v,
      repeats = outer_repeats,
      strata = c("Qmax")
    ),
    inside = vfold_cv(
      v = inner_v,
      repeats = inner_repeats,
      strata = c("Qmax")
    )
  )
  
  for (outer_i in 1:(outer_repeats * outer_v)) {
    c(dtrain, dtest, val, train_index, test_index) %<-%
      prepare_train_test_data(
        outer_i,
        data_process,
        cv_folds,
        xgboost_format = T,
        return_index = T
      )
    
    optObj <- bayesOpt(
      FUN = scoringFunction,
      bounds = bounds,
      initPoints = initPoints,
      iters.n = 1,
      iters.k = 1,
      plotProgress = plotProgress
    )
    
    for (iter in (initPoints + 1):max_iter) {
      optObj <- updateGP(optObj)
      optObj <- addIterations(optObj, iters.n = 1, iters.k = 1)
      if ((iter - which.max(optObj$scoreSummary$Score)) > patience &&
          iter > (patience + initPoints)) {
        # early stop if no improvement > patience, and iter > patience + initPoints
        break
      }
    }
    
    temp %<-% final_xgb_model(optObj, region, season, outer_i)
    
    save(
      cv_folds,
      file = paste0(
        "./modeling_results/xgb_region",
        region,
        "_",
        season,
        "_iter",
        outer_i,
        "_cv.Rda"
      )
    )
  }
}

end_time <- Sys.time()
end_time - start_time
