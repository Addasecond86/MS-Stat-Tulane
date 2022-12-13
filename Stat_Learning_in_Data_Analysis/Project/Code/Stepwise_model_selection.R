library(tidyverse)
library(lubridate)
library(MASS)

set.seed(20221107)

# Load data
data <-
  read_csv(
    "/Users/zehao/Documents/GitHub/MS-Stat-Tulane/Stat_Learning_in_Data_Analysis/Project/data/weatherAUS.csv"
  )

# Drop rows with missing values
data <- data %>%
  drop_na()

sapply(lapply(data, unique), length)

# data["Location"] <- factor(data$Location)

data[c("RainTomorrow")][data[c("RainTomorrow")] == "No"] <-
  "0"
data[c("RainTomorrow")][data[c("RainTomorrow")] == "Yes"] <-
  "1"

data$RainTomorrow <- as.integer(data$RainTomorrow)

# split data into training and testing
data <- data[, 2:23]

data %>% filter(RainTomorrow==1) %>% 
  count()

data <- rbind(slice(filter(data, RainTomorrow == 0), sample(c(1:43993), 12427)),
      filter(data, RainTomorrow == 1))

sample_ind <- sample(c(1:dim(data)[1]), dim(data)[1] * 0.8)
data_train <- data %>%
  slice(sample_ind)

sapply(lapply(data_train, unique), length)

data_test <- data %>%
  slice(-sample_ind)

sapply(lapply(data_test, unique), length)

# levels(data_train$Location)

# Full Model
full <- glm(RainTomorrow ~ .,
            family = binomial(link = "logit"),
            data = data)

# Stepwise model selection
stepwise <- step(full,
                 direction = 'both',
                 scope = formula(full))
mean(round(predict(stepwise, newdata = data_test[,-22], type = c('response')))==data_test[,22])

