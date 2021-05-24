

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
      dtrain <-
        xgb.DMatrix(data = data.matrix(training_outer %>% select(-Qmax, -EventID)),
                    label = training_outer$Qmax)
      dtest <-
        xgb.DMatrix(data = data.matrix(test %>% select(-Qmax, -EventID)),
                    label = test$Qmax)
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

train_model <-
  function(method,
           tuneLength = NULL,
           tuneGrid = NULL,
           recipe,
           dtrain,
           ctrl) {
    if (all(is.null(tuneLength), is.null(tuneGrid))) {
      return(NULL)
    }
    
    start_time <- Sys.time()
    
    cl <- makePSOCKcluster(detectCores() - 2)
    registerDoParallel(cl)
    
    if (!is.null(tuneLength)) {
      ModelFit <- caret::train(
        recipe,
        dtrain,
        method = method,
        tuneLength = tuneLength,
        trControl = ctrl
      )
    } else {
      ModelFit <- caret::train(
        recipe,
        dtrain,
        method = method,
        tuneGrid = tuneGrid,
        trControl = ctrl
      )
    }
    
    stopCluster(cl)
    
    cat(difftime(Sys.time(), start_time, units = 'mins'), "\n")
    
    ModelFit
  }

compare_models <- function(model_list) {
  # return the model with lowest RMSE; for comparing different preprocessing procedures
  outIndex <-
    lapply(model_list, function(x)
      x$results$RMSE %>% min) %>%
    which.min()
  
  model_list[[outIndex]]
}

train_all_models <-
  function(dtrain,
           dtest,
           val,
           train_index,
           test_index) {
    # create data containing features
    
    # preprocess using recipe
    recipe_no_pca <- recipe(Qmax ~ ., data = dtrain) %>%
      step_YeoJohnson(-matches('(X[0-9]$)|Qmax')) %>%
      step_center(-matches('(X[0-9]$)|Qmax')) %>%
      step_scale(-matches('(X[0-9]$)|Qmax'))
    
    recipe_pca <- recipe(Qmax ~ ., data = dtrain) %>%
      step_YeoJohnson(-matches('(X[0-9]$)|Qmax')) %>%
      step_center(-matches('(X[0-9]$)|Qmax')) %>%
      step_scale(-matches('(X[0-9]$)|Qmax')) %>%
      step_pca(-matches('(X[0-9]$)|Qmax'), threshold = 0.95)
    
    recipe_org <- recipe(Qmax ~ ., data = dtrain)
    
    # Train control
    ctrl <- trainControl(
      method = "cv",
      number = length(val),
      index = val,
      savePredictions = "none",
      returnData = FALSE
    )
    
    # Start training
    # 1
    cat("LM starts\n")
    
    lmFit <-
      train_model("lm",
                  tuneLength = 1,
                  tuneGrid = NULL,
                  recipe_no_pca,
                  dtrain,
                  ctrl)
    
    lmFit1 <-
      train_model("lm",
                  tuneLength = 1,
                  tuneGrid = NULL,
                  recipe_pca,
                  dtrain,
                  ctrl)
    
    lmFit2 <-
      train_model("lm",
                  tuneLength = 1,
                  tuneGrid = NULL,
                  recipe_org,
                  dtrain,
                  ctrl)
    
    lmFit <-
      compare_models(list(lmFit, lmFit1, lmFit2)) # rename the best model to xxxFit
    rm(lmFit1, lmFit2) # remove the other Fits
    
    
    # 2
    cat("MARS starts\n")
    
    MARSFit <-
      train_model(
        "bagEarth",
        tuneLength = 5,
        tuneGrid = NULL,
        recipe_no_pca,
        dtrain,
        ctrl
      )
    
    MARSFit1 <-
      train_model("bagEarth",
                  tuneLength = 5,
                  tuneGrid = NULL,
                  recipe_pca,
                  dtrain,
                  ctrl)
    
    MARSFit2 <-
      train_model("bagEarth",
                  tuneLength = 5,
                  tuneGrid = NULL,
                  recipe_org,
                  dtrain,
                  ctrl)
    
    MARSFit <- compare_models(list(MARSFit, MARSFit1, MARSFit2))
    rm(MARSFit1, MARSFit2)
    
    
    # 3
    cat("PLS starts\n")
    
    PLSFit <-
      train_model("pls",
                  tuneLength = 5,
                  tuneGrid = NULL,
                  recipe_no_pca,
                  dtrain,
                  ctrl)
    
    PLSFit1 <-
      train_model("pls",
                  tuneLength = 5,
                  tuneGrid = NULL,
                  recipe_pca,
                  dtrain,
                  ctrl)
    
    PLSFit2 <-
      train_model("pls",
                  tuneLength = 5,
                  tuneGrid = NULL,
                  recipe_org,
                  dtrain,
                  ctrl)
    
    PLSFit <- compare_models(list(PLSFit, PLSFit1, PLSFit2))
    rm(PLSFit1, PLSFit2)
    
    
    # 4
    cat("Ridge starts\n")
    
    RidgeFit <-
      train_model("ridge",
                  tuneLength = 5,
                  tuneGrid = NULL,
                  recipe_no_pca,
                  dtrain,
                  ctrl)
    
    RidgeFit1 <-
      train_model("ridge",
                  tuneLength = 5,
                  tuneGrid = NULL,
                  recipe_pca,
                  dtrain,
                  ctrl)
    
    RidgeFit2 <-
      train_model("ridge",
                  tuneLength = 5,
                  tuneGrid = NULL,
                  recipe_org,
                  dtrain,
                  ctrl)
    
    RidgeFit <- compare_models(list(RidgeFit, RidgeFit1, RidgeFit2))
    rm(RidgeFit1, RidgeFit2)
    
    
    # 5
    cat("Lasso starts\n")
    
    LassoFit <-
      train_model("lasso",
                  tuneLength = 5,
                  tuneGrid = NULL,
                  recipe_no_pca,
                  dtrain,
                  ctrl)
    
    LassoFit1 <-
      train_model("lasso",
                  tuneLength = 5,
                  tuneGrid = NULL,
                  recipe_pca,
                  dtrain,
                  ctrl)
    
    LassoFit2 <-
      train_model("lasso",
                  tuneLength = 5,
                  tuneGrid = NULL,
                  recipe_org,
                  dtrain,
                  ctrl)
    
    LassoFit <- compare_models(list(LassoFit, LassoFit1, LassoFit2))
    rm(LassoFit1, LassoFit2)
    
    
    # 6
    cat("CART starts\n")
    
    CARTFit <-
      train_model(
        "treebag",
        tuneLength = 5,
        tuneGrid = NULL,
        recipe_no_pca,
        dtrain,
        ctrl
      )
    
    CARTFit1 <-
      train_model("treebag",
                  tuneLength = 5,
                  tuneGrid = NULL,
                  recipe_pca,
                  dtrain,
                  ctrl)
    
    CARTFit2 <-
      train_model("treebag",
                  tuneLength = 5,
                  tuneGrid = NULL,
                  recipe_org,
                  dtrain,
                  ctrl)
    
    CARTFit <- compare_models(list(CARTFit, CARTFit1, CARTFit2))
    rm(CARTFit1, CARTFit2)
    
    
    # 7
    cat("KNN starts\n")
    
    KNNGrid <- expand.grid(k = c(1, 3, 5, 7, 9, 11))
    
    KNNFit <-
      train_model("knn",
                  tuneLength = NULL,
                  tuneGrid = KNNGrid,
                  recipe_no_pca,
                  dtrain,
                  ctrl)
    
    KNNFit1 <-
      train_model("knn",
                  tuneLength = NULL,
                  tuneGrid = KNNGrid,
                  recipe_pca,
                  dtrain,
                  ctrl)
    
    KNNFit2 <-
      train_model("knn",
                  tuneLength = NULL,
                  tuneGrid = KNNGrid,
                  recipe_org,
                  dtrain,
                  ctrl)
    
    KNNFit <- compare_models(list(KNNFit, KNNFit1, KNNFit2))
    rm(KNNFit1, KNNFit2)
    
    
    # 8
    cat("Cubist starts\n")
    
    CubistGrid <-
      expand.grid(committees = c(1, 5, 10, 25, 50, 75, 100),
                  neighbors = c(0, 1, 3, 5, 7))
    
    CubistFit <-
      train_model("cubist",
                  tuneLength = NULL,
                  tuneGrid = CubistGrid,
                  recipe_no_pca,
                  dtrain,
                  ctrl)
    
    CubistFit1 <-
      train_model("cubist",
                  tuneLength = NULL,
                  tuneGrid = CubistGrid,
                  recipe_pca,
                  dtrain,
                  ctrl)
    
    CubistFit2 <-
      train_model("cubist",
                  tuneLength = NULL,
                  tuneGrid = CubistGrid,
                  recipe_org,
                  dtrain,
                  ctrl)
    
    
    CubistFit <- compare_models(list(CubistFit, CubistFit1, CubistFit2))
    rm(CubistFit1, CubistFit2)
    
    
    # 9
    cat("SVMPoly starts\n")
    
    svmGrid <- expand.grid(
      degree = c(1, 2),
      scale = c(0.001, 0.002, 0.005, 0.01, 0.1),
      C = 2 ^ (-2:6)
    )
    
    svmGrid <- svmGrid[sample(x = 1:nrow(svmGrid), size = 30), ]
    
    SVMFit <-
      train_model(
        "svmPoly",
        tuneLength = NULL,
        tuneGrid = svmGrid,
        recipe_no_pca,
        dtrain,
        ctrl
      )
    
    SVMFit1 <-
      train_model("svmPoly",
                  tuneLength = NULL,
                  tuneGrid = svmGrid,
                  recipe_pca,
                  dtrain,
                  ctrl)
    
    SVMFit2 <-
      train_model("svmPoly",
                  tuneLength = NULL,
                  tuneGrid = svmGrid,
                  recipe_org,
                  dtrain,
                  ctrl)
    
    SVMFit <- compare_models(list(SVMFit, SVMFit1, SVMFit2))
    rm(SVMFit1, SVMFit2)
    
    
    # 10
    cat("svmRadial starts\n")
    
    svmRadialFit <-
      train_model(
        "svmRadial",
        tuneLength = 10,
        tuneGrid = NULL,
        recipe_no_pca,
        dtrain,
        ctrl
      )
    
    svmRadialFit1 <-
      train_model("svmRadial",
                  tuneLength = 10,
                  tuneGrid = NULL,
                  recipe_pca,
                  dtrain,
                  ctrl)
    
    svmRadialFit2 <-
      train_model("svmRadial",
                  tuneLength = 10,
                  tuneGrid = NULL,
                  recipe_org,
                  dtrain,
                  ctrl)
    
    svmRadialFit <- compare_models(list(svmRadialFit, svmRadialFit1, svmRadialFit2))
    rm(svmRadialFit1, svmRadialFit2)
    
    
    # 11
    cat("RF starts\n")
    
    RFFit <-
      train_model("rf",
                  tuneLength = 10,
                  tuneGrid = NULL,
                  recipe_no_pca,
                  dtrain,
                  ctrl)
    
    RFFit1 <-
      train_model("rf",
                  tuneLength = 10,
                  tuneGrid = NULL,
                  recipe_pca,
                  dtrain,
                  ctrl)
    
    RFFit2 <-
      train_model("rf",
                  tuneLength = 10,
                  tuneGrid = NULL,
                  recipe_org,
                  dtrain,
                  ctrl)
    
    RFFit <- compare_models(list(RFFit, RFFit1, RFFit2))
    rm(RFFit1, RFFit2)
    
    # output
    models <- list(
      lmFit = lmFit,
      # 1
      MARSFit = MARSFit,
      # 2
      PLSFit = PLSFit,
      # 3
      RidgeFit = RidgeFit,
      # 4
      LassoFit = LassoFit,
      # 5
      CARTFit = CARTFit,
      # 6
      KNNFit = KNNFit,
      # 7
      CubistFit = CubistFit,
      # 8
      SVMFit = SVMFit,
      # 9
      svmRadialFit = svmRadialFit,
      # 10
      RFFit = RFFit # 11
    )
    
    models
  }

run_models_outer_CV <-
  function(outer_i,
           data_process,
           cv_folds,
           return_index = T) {
    c(dtrain, dtest, val, train_index, test_index) %<-%
      prepare_train_test_data(
        outer_i,
        data_process,
        cv_folds,
        xgboost_format = F,
        return_index = return_index
      )
    
    models <-
      train_all_models(dtrain, dtest, val, train_index, test_index)
    
    list(models = models,
         dtrain = dtrain,
         dtest = dtest)
  }

run_models_files <- function(region, summer_winter) {
  set.seed(SEED)
  
  # read data and preprocess
  file_names <- list(
    paste0(
      "./raw_data/5_Reg",
      region,
      "_Tr_",
      summer_winter,
      ".csv"
    ),
    paste0(
      "./raw_data/5_Reg",
      region,
      "_Te_",
      summer_winter,
      ".csv"
    )
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
  
  # run outer CV iterations
  for (outer_i in 1:(outer_repeats * outer_v)) {
    cat("Outer iteration ", outer_i, " starts", "\n")
    
    c(models, dtrain, dtest) %<-% run_models_outer_CV(outer_i, data_process, cv_folds)
    
    save(
      models,
      dtrain,
      dtest,
      cv_folds,
      file = paste0(
        "./modeling_results/Reg",
        region,
        "_",
        summer_winter,
        "_iter",
        outer_i,
        ".Rda"
      )
    )
    
    cat("Outer iteration ", outer_i, " ends", "\n")
  }
}

# Evaluation --------------------------------------------------------------

SEED = 439759
outer_repeats = 1
inner_repeats = 1
outer_v = 5
inner_v = 5

input_df <- expand_grid(region = c(1:4),
                        season = c("S", "W"))

start_time <- Sys.time()
for (i in 1:nrow(input_df)) {
  region <- input_df$region[i]
  season <- input_df$season[i]
  
  run_models_files(region = region, summer_winter = season)
}
end_time <- Sys.time()
end_time - start_time
