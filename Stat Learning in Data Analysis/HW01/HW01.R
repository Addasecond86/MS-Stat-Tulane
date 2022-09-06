library("tidyverse")
library("class")

set.seed(0906)

gene_data <-
  read_table(
    "/Users/zehao/Documents/GitHub/MS-Stat-Tulane/Stat Learning in Data Analysis/HW01/HW1data.txt",
    col_names = FALSE
  )
# X2 is Y

# Randomly slelect 10 columns

gene_10 <- select(gene_data, sample(c(3:6832), 10))

# Use 62.5% as train data

train_data_sample <- sample(c(1:64), 40)

train_data <- slice(gene_10, train_data_sample)

test_data <- slice(gene_10,-train_data_sample)

train_data_label <- pull(slice(gene_data, train_data_sample), 2)

test_data_label <- pull(slice(gene_data,-train_data_sample), 2)

knn_5 <-
  as.integer(as.character(
    knn(
      train = train_data,
      test = test_data,
      cl = train_data_label,
      k = 5
    )
  ))

# Accuracy

Err_5 <- mean(abs(knn_5 - test_data_label))

Acc_5 <- 1 - Err_5


knn_2 <-
  as.integer(as.character(
    knn(
      train = train_data,
      test = test_data,
      cl = train_data_label,
      k = 2
    )
  ))

Err_2 <- mean(abs(knn_2 - test_data_label))

Acc_2 <- 1 - Err_2

knn_15 <-
  as.integer(as.character(
    knn(
      train = train_data,
      test = test_data,
      cl = train_data_label,
      k = 15
    )
  ))

Err_15 <-
  mean(abs(knn_15 - test_data_label))

Acc_15 <- 1 - Err_15

# Plot

tibble(
  k = c(2, 5, 15),
  Acc = c(Acc_2, Acc_5, Acc_15),
  Err = c(Err_2, Err_5, Err_15)
) %>% ggplot() +
  geom_line(mapping = aes(k, Acc, color = "Accuracy"),
            linetype = "longdash") +
  geom_line(mapping = aes(k, Err, color = "Error")) +
  theme_light() +
  scale_x_continuous(limits = c(1, 16), breaks = seq(2, 16, 2)) +
  ylab("Accuracy or Error") +
  labs(color = "Type")
