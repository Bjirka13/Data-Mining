install.packages('PerformanceAnalytics')

library(tidyverse)  # data manipulation
library(cluster)    # clustering algorithms
library(factoextra) # clustering algorithms & visualization
library(PerformanceAnalytics)
library(ggpubr)
library(tibble)
library(MVN)

# Load dataset
data = read.csv(file.choose(),header=TRUE)

# EDA
str(data)
cat("\nMissing value:", sum(is.na(data)), "\n")
cat("Data duplikat:", sum(duplicated(data)), "\n")

# Label Encoding untuk semua kolom kategorikal (character/factor)
data_encoded <- data %>%
  mutate(across(where(~ is.character(.x) || is.factor(.x)),
                ~ as.numeric(factor(.x))))

# Scaling untuk semua kolom numerik hasil encoding
scaled_data <- scale(data_encoded)

# Elbow method
wss <- sapply(1:10, function(k) {
  kmeans(scaled_data, centers = k, nstart = 20)$tot.withinss
})

# Train K-Means (misal k = 3)
set.seed(42)
model <- kmeans(scaled_data, centers = 3, nstart = 25)

# Tambahkan cluster ke dataset asli
output <- data %>% mutate(cluster = model$cluster)

# Simpan hasil clustering
write_csv(output, "clustering_output.csv")
