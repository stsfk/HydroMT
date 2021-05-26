if (!require("pacman"))
  install.packages("pacman")
pacman::p_load(
  tidyverse,
  tidymodels,
  caret,
  lubridate,
  zeallot,
  xgboost,
  randomForest,
  ParBayesianOptimization,
  XML,
  curl,
  keras
)


# Setup -------------------------------------------------------------------


set.seed(4)
dir_path <- c("./modeling_results/lstm/")

if (!dir.exists(dir_path)) {
  dir.create(dir_path)
}

# Prepare data ------------------------------------------------------------


URL <- "https://hydrology.nws.noaa.gov/pub/gcip/mopex/US_Data/Us_438_Daily/"
fnames <- curl(URL) %>%
  readLines() %>%
  getHTMLLinks() %>%
  .[str_detect(., "dly$")]

download_data <- function(fname) {
  url <- paste(URL, fname, sep = "")
  download.file(url,
                destfile = paste0("./modeling_results/lstm/", fname))
  
  fname
}

read_data <- function(fname) {
  read_table(
    paste0("./modeling_results/lstm/", fname),
    col_names = c("date", "pp", "pet", "r", "Tmax", "Tmin"),
    cols(
      date = col_character(),
      pp = col_double(),
      pet = col_double(),
      r = col_double(),
      Tmax = col_double(),
      Tmin = col_double()
    )
  ) %>%
    dplyr::mutate(
      year = str_sub(date, 1, 4),
      month = str_sub(date, 5, 6),
      date = str_sub(date, 7, 8),
      date = paste(year, month, date, sep = "-"),
      date = ymd(date)
    ) %>%
    dplyr::select(date, r, pp, pet, Tmax, Tmin)
}



# Setup modeling ----------------------------------------------------------


case_tested <- sample(fnames, 10)

fname <- case_tested[[1]]

data_raw <- fname %>%
  download_data %>%
  read_data()

data_process <- data_raw %>%
  dplyr::filter(pp >= 0) %>%
  dplyr::select(-date)

training <- data_process[1:(365 * 20), ] # 20 years
test <- data_process[((365 * 30) + 1):((365 * 32) + 1), ]

divide_data <- function(df, n_steps = 365, x_dim = 4) {
  # Input shape: batch_size, time_step, feature
  # Output shape: batch_size, response(1d)
  df <- df[, c("r", "pp", "pet", "Tmax", "Tmin")]
  
  n_batch <- nrow(df) - 364
  x <- array(0, dim = c(n_batch, n_steps, x_dim))
  y <- matrix(0, nrow = n_batch, ncol = 1)
  
  for (i in 1:n_batch) {
    s_ind <- i
    e_ind <- i + n_steps - 1
    
    x[i, seq_along(s_ind:e_ind), ] <-
      data.matrix(df[s_ind:e_ind, c(2:(1 + x_dim))])
    y[i,] <- data.matrix(df[e_ind, 1])
  }
  
  return(list(x = x,
              y = y))
}

N_STEP = 365
c(x_train, y_train) %<-% divide_data(training, n_steps = N_STEP)
c(x_test, y_test) %<-% divide_data(test, n_steps = N_STEP)


# LSTM functions ----------------------------------------------------------

N_EPOCH <- 30

scoringFunction <-
  function(HIDDEN_DIM1,
           HIDDEN_DIM2,
           dense_unit1,
           dropout_rate,
           dense_unit2,
           BATCH_SIZE) {
    model <- keras_model_sequential() %>%
      layer_lstm(
        units = HIDDEN_DIM1,
        input_shape = c(N_STEP, 4),
        return_sequences = T
      ) %>%
      layer_lstm(units = HIDDEN_DIM2,
                 return_sequences = F) %>%
      layer_dense(units = dense_unit1, activation = "relu") %>%
      layer_dropout(rate = dropout_rate) %>%
      layer_dense(units = dense_unit2, activation = "relu") %>%
      layer_dense(units = 1)
    
    model %>% compile(
      loss = 'mse',
      optimizer = 'adam',
      metrics = list("mean_absolute_error")
    )
    
    n_h5_files <- dir("./modeling_results/lstm/", "*.h5") %>%
      length()
    
    new_h5_fname <-
      paste0("./modeling_results/lstm/check_point", n_h5_files + 1, ".h5")
    
    history <- model %>% fit(
      x_train,
      y_train,
      epochs = N_EPOCH,
      batch_size = BATCH_SIZE,
      validation_split = 0.2,
      shuffle = T,
      callbacks = list(
        callback_model_checkpoint(filepath = new_h5_fname,
                                  save_best_only = T),
        callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1)
      )
    )
    
    score <- history$metrics$val_loss %>%
      min()
    
    gc()
    
    list(Score = -score)
  }


clean_h5_files <- function() {
  delfiles <- dir("./modeling_results/lstm/" , pattern = "*.h5")
  unlink(file.path("./modeling_results/lstm/", delfiles))
}


# Optimization ------------------------------------------------------------


bounds <- list(
  HIDDEN_DIM1 = c(4L, 64L),
  HIDDEN_DIM2 = c(4L, 64L),
  dense_unit1 = c(4L, 64L),
  dropout_rate = c(0, 0.5),
  dense_unit2 = c(2L, 8L),
  BATCH_SIZE = c(8L, 256L)
)

clean_h5_files()
optObj <- bayesOpt(
  FUN = scoringFunction,
  bounds = bounds,
  initPoints = 10,
  iters.n = 10,
  iters.k = 1,
  plotProgress = T
)




# Gof ---------------------------------------------------------------------


load("./modeling_results/lstm/optObj.Rda")

opt_id <- which.max(optObj$scoreSummary$Score)
model <- load_model_hdf5(paste0("./modeling_results/lstm/check_point", opt_id, ".h5"))

create_lstm_input <- function(row_id, df = data_process) {
  row_id <-
    c((row_id[1] - 364):last(row_id)) # to include the antecedent time series to support
  out <- df[row_id, ]
  
  divide_data(out, n_steps = N_STEP)
}

data_eval <- data_process %>%
  mutate(id = 1:n(),
         datetime = data_raw$date) %>%
  select(id, datetime, everything()) %>%
  dplyr::filter(datetime >= ymd("1973-10-01"),
                datetime <= ymd("1974-09-30"))

row_id_of_interest <- data_eval$id

c(x_interest, y_interest) %<-% create_lstm_input(row_id_of_interest)


pred <- predict(model, x_interest) %>%
  as.vector()

ob <- y_interest %>%
  as.vector()

model_gof <- hydroGOF::gof(pred, ob)

data_plot <- tibble(date = data_eval$datetime,
                    observation = ob,
                    lstm = pred) %>%
  gather(case, value,-date)


ggplot(data_plot,
       aes(
         date,
         value,
         color = case,
         line_type = case,
         linetype = case
       )) +
  scale_x_date(date_labels = "%b %d") +
  geom_line() +
  annotate(
    "text",
    x = ymd("1973-10-15"),
    y = 11,
    label = paste0("r-squared = ", model_gof[17], "\nNSE = ", model_gof[9]),
    hjust = 0
  ) +
  labs(y = "Streamflow discharge [mm]") +
  theme_bw() +
  theme(
    legend.title = element_blank(),
    legend.position = c(0.99, 0.99),
    legend.justification = c(1, 1),
    legend.background = element_blank(),
    axis.title.x = element_blank()
  )


ggsave(
  filename = "./paper_figures/Figure_S11.svg",
  width = 6,
  height = 4,
  units = "in",
  dpi = 600
)




# PP ----------------------------------------------------------------------

change_pp <- function(ratio) {
  c(x_interest, y_interest) %<-% create_lstm_input(row_id_of_interest, data_raw %>%
                                                     mutate(pp = pp * ratio))
  
  pred <- predict(model, x_interest) %>%
    as.vector()
  
  out <- tibble(date = data_eval$datetime,
                lstm = pred,
                ratio = ratio)
  
  out
}

ratios <- (8:12)*0.1
data_plot_pps <- vector("list", length = length(ratios))
for (i in seq_along(ratios)) {
  data_plot_pps[[i]] <- change_pp(ratio = ratios[[i]])
}

data_plot_pp <- data_plot_pps %>%
  bind_rows() %>%
  mutate(
    ratio = ratio*100,
    ratio = factor(ratio))

ggplot(data_plot_pp, aes(date, lstm, color = ratio)) +
  geom_line() +
  scale_color_viridis(discrete = TRUE) +
  scale_x_date(date_labels = "%b %d") +
  labs(y = "Streamflow discharge [mm]") +
  theme_bw() +
  labs(color = "Magnitude of precipitation\ncompared to the original value [%]")+
  theme(
    legend.position = c(0.99, 0.99),
    legend.justification = c(1, 1),
    legend.background = element_blank(),
    axis.title.x = element_blank()
  )

ggsave(
  filename = "./paper_figures/Figure_S12.png",
  width = 7,
  height = 5,
  units = "in",
  dpi = 600,
)

data_plot_pp %>%
  group_by(ratio) %>%
  summarise(runoff_volume = sum(lstm)) %>%
  view()


# PET ---------------------------------------------------------------------


change_pet <- function(ratio) {
  c(x_interest, y_interest) %<-% create_lstm_input(row_id_of_interest, data_raw %>%
                                                     mutate(pet = pet * ratio))
  
  pred <- predict(model, x_interest) %>%
    as.vector()
  
  out <- tibble(date = data_eval$datetime,
                lstm = pred,
                ratio = ratio)
  
  out
}

ratios <- (8:12)*0.1
data_plot_pets <- vector("list", length = length(ratios))
for (i in seq_along(ratios)) {
  data_plot_pets[[i]] <- change_pet(ratio = ratios[[i]])
}

data_plot_pet <- data_plot_pets %>%
  bind_rows() %>%
  mutate(
    ratio = ratio*100,
    ratio = factor(ratio))

ggplot(data_plot_pet, aes(date, lstm, color = ratio)) +
  geom_line() +
  scale_color_viridis(discrete = TRUE) +
  scale_x_date(date_labels = "%b %d") +
  labs(y = "Streamflow discharge [mm]") +
  theme_bw() +
  labs(color = "Magnitude of potential evaporation\ncompared to the original value [%]")+
  theme(
    legend.position = c(0.99, 0.99),
    legend.justification = c(1, 1),
    legend.background = element_blank(),
    axis.title.x = element_blank()
  )

ggsave(
  filename = "./paper_figures/Figure_S13.png",
  width = 7,
  height = 5,
  units = "in",
  dpi = 600,
)

data_plot_pet %>%
  group_by(ratio) %>%
  summarise(runoff_volume = sum(lstm)) %>%
  view()



# T experiment ------------------------------------------------------------

change_t <- function(Tchange) {
  
  df <- data_raw %>%
    mutate(Tmax = Tmax + Tchange*(date>=ymd("1974-03-01")),
           Tmin = Tmin + Tchange*(date>=ymd("1974-03-01")))
  
  c(x_interest, y_interest) %<-% create_lstm_input(row_id_of_interest, df)
  
  pred <- predict(model, x_interest) %>%
    as.vector()
  
  out <- tibble(date = data_eval$datetime,
                lstm = pred,
                Tchange = Tchange)
  
  out
}

Tchanges <- -2:2
data_plot_ts <- vector("list", length = length(Tchanges))
for (i in seq_along(ratios)) {
  data_plot_ts[[i]] <- change_t(Tchange = Tchanges[[i]])
}

data_plot_t <- data_plot_ts %>%
  bind_rows() %>%
  mutate(
    Tchange = factor(Tchange))

ggplot(data_plot_t %>% filter(date>=ymd("1974-03-01")), aes(date, lstm, color = Tchange)) +
  geom_line() +
  scale_color_viridis(discrete = TRUE) +
  scale_x_date(date_labels = "%b %d") +
  labs(y = "Streamflow discharge [mm]") +
  theme_bw() +
  labs(color = "Changes in daily maximum/minimum\nair temperature after March 01 [Celsius]")+
  theme(
    legend.position = c(0.99, 0.99),
    legend.justification = c(1, 1),
    legend.background = element_blank(),
    axis.title.x = element_blank()
  )

ggsave(
  filename = "./paper_figures/Figure_S14.png",
  width = 7,
  height = 5,
  units = "in",
  dpi = 600,
)

data_plot_t %>% filter(date >= ymd("1974-03-01")) %>%
  group_by(Tchange) %>%
  summarise(runoff_volume = sum(lstm)) %>%
  view()





# Recycle -----------------------------------------------------------------


