library("tidyverse")
library("class")

set.seed(0927)

gene_data <-
  read_table(
    "/Users/zehao/Documents/GitHub/MS-Stat-Tulane/Stat Learning in Data Analysis/HW02/HW1data.txt",
    col_names = FALSE
  )
# X2 is Y

p <- c()

Y <- pull(gene_data, 2)

for (i in 3:6832) {
  X <- pull(gene_data, i)
  single_p <-
    summary(glm(Y ~ X, family = binomial(link = "logit")))$coefficients[2, 4]
  p = append(p, single_p)
}

gene_10 <- gene_data[, (order(p)[1:10] + 2)]

# Use all data as train data

train_data <- gene_10

train_data_label <- Y

knn_15 <-
  as.integer(as.character(
    knn(
      train = train_data,
      test = train_data,
      cl = train_data_label,
      k = 15
    )
  ))

Err_15 <- mean(abs(knn_15 - train_data_label))

logi <-
  glm(Y ~ .,
      family = binomial(link = "logit"),
      data = cbind(Y, gene_10))

predict_result <-
  round(predict.glm(logi, gene_10, type = "response"), 2)

mean(abs(Y - predict_result))

library(MASS)

lda_result <-
  lda(Y ~ ., data = cbind(Y, gene_10))
lda_result <-
  predict(lda_result, gene_10)$class

mean(abs(Y - (as.numeric(lda_result) - 1)))
