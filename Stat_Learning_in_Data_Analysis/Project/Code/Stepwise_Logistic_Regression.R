library(tidyverse)
library(lubridate)
library(MASS)
data <-
  read_csv(
    "/Users/zehao/Documents/GitHub/MS-Stat-Tulane/Stat Learning in Data Analysis/Project/data/weatherAUS.csv"
  )

data <- data %>%
  drop_na('RainToday', 'RainTomorrow', 'Location')

sapply(lapply(data, unique), length)

data['Location'] <- factor(data$Location)


data[c('RainToday', 'RainTomorrow')][data[c('RainToday', 'RainTomorrow')] == "No"] <-
  "0"
data[c('RainToday', 'RainTomorrow')][data[c('RainToday', 'RainTomorrow')] == "Yes"] <-
  "1"

data$RainToday <- as.integer(data$RainToday)
data$RainTomorrow <- as.integer(data$RainTomorrow)

data <- data %>%
  mutate(year = year(Date), month = month(Date))

fillNA <- data[, c(1, 3:7, 9, 12:21)]
fillNA[is.na(fillNA)] <- 0

fillNA <- fillNA %>%
  mutate(year = year(Date), month = month(Date)) %>%
  group_by(year, month) %>%
  summarise(
    MinTemp = median(MinTemp),
    MaxTemp = median(MaxTemp),
    Rainfall = median(Rainfall),
    Evaporation = median(Evaporation),
    Sunshine = median(Sunshine),
    WindGustSpeed = median(WindGustSpeed),
    WindSpeed9am = median(WindSpeed9am),
    WindSpeed3pm = median(WindSpeed3pm),
    Humidity9am = median(Humidity9am),
    Humidity3pm = median(Humidity3pm),
    Pressure9am = median(Pressure9am),
    Pressure3pm = median(Pressure3pm),
    Cloud9am = median(Cloud9am),
    Cloud3pm = median(Cloud3pm),
    Temp9am = median(Temp9am),
    Temp3pm = median(Temp3pm)
  )

data <- data %>%
  left_join(fillNA, by = c("year" = "year", "month" = "month"))

name_x <- names(data)[c(3:7, 9, 12:21)]
name_y <- names(data)[c(26:41)]

for (i in 1:16) {
  ind <- is.na(data[name_x[i]])[, 1]
  data[name_x[i]][ind, ] <- pull(data[name_y[i]])[ind]
}

data <- data[, 2:23]
sample_ind <- sample(c(1:dim(data)[1]), dim(data)[1] * 0.8)
data_train <- data %>%
  slice(sample_ind)

sapply(lapply(data_train, unique), length)

data_test <- data %>%
  slice(-sample_ind)

sapply(lapply(data_test, unique), length)

levels(data_train$Location)

null <- glm(RainTomorrow ~ 1,
            family = binomial(link = "logit"),
            data = data)
full <-  glm(RainTomorrow ~ .,
             family = binomial(link = "logit"),
             data = data)
step(
  full,
  scope = list(lower = null, upper = full),
  direction = "both",
  criterion = "AIC"
)
