# ============================================================
# 1. LOAD LIBRARIES
# ============================================================
library(readr)
library(dplyr)
library(ggplot2)
library(cluster)
library(caret)
library(e1071)
library(rpart)
library(rpart.plot)
library(randomForest)
library(arules)
library(arulesViz)
library(factoextra)
library(reshape2)

# ============================================================
# 2. LOAD DATA
# ============================================================
setwd("E:\\Kuliah\\Tugas\\SMT 5\\Data Mining\\Dataset")
df <- read_csv("StudentPerformanceFactors.csv")

# ============================================================
# 3. EDA
# ============================================================
cat("Struktur Dataset")
glimpse(df)

cat("Statistik Deskriptif")
summary(df)

cat("Jumlah Missing Value")
print(sum(is.na(df)))
print(colSums(is.na(df)))

cat("Jumlah Duplicate Data")
print(sum(duplicated(df)))

cat("HISTOGRAM DISTRIBUSI SETIAP FITUR NUMERIK\n")
numeric_df <- df %>% select(where(is.numeric))
numeric_long <- melt(numeric_df)

ggplot(numeric_long, aes(x = value)) +
  geom_histogram(bins = 30, fill = "steelblue",
                 color = "white", alpha = 0.8) +
  facet_wrap(~ variable, scales = "free", ncol = 3) +
  theme_minimal(base_size = 14) +
  labs(title = "Distribusi Histogram Fitur Numerik",
       x = "Value", y = "Frequency") +
  theme(panel.grid = element_blank())

# DEteksi Outlier dengan IQR
cat("OUTLIER DETECTION (IQR METHOD)\n")
detect_outliers <- function(x) {
  Q1 <- quantile(x, 0.25, na.rm = TRUE)
  Q3 <- quantile(x, 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  lower <- Q1 - 1.5 * IQR
  upper <- Q3 + 1.5 * IQR
  sum(x < lower | x > upper)
}

outlier_counts <- sapply(numeric_df, detect_outliers)
print(outlier_counts)


cat("CORRELATION HEATMAP\n")
corr_matrix <- cor(numeric_df, use = "complete.obs")
corr_long <- melt(corr_matrix, varnames = c("Var1", "Var2"),
                  value.name = "Correlation")

ggplot(corr_long, aes(Var1, Var2, fill = Correlation)) +
  geom_tile(color = "white") +
  geom_text(aes(label = round(Correlation, 2)), size = 5) +
  scale_fill_gradient2(low = "blue", high = "red",
                       mid = "white", midpoint = 0,
                       limit = c(-1, 1)) +
  theme_minimal(base_size = 14) +
  labs(title = "Correlation Matrix (ggplot2)",
       x = "", y = "") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        panel.grid = element_blank())

# ============================================================
# 3. DATA CLEANING
# ============================================================
# Handling Noise
df <- df %>% filter(Exam_Score <= 100)

# Handling Duplicate Data
df <- df %>% distinct()

# Deklarasi Fungsi Mode untuk pengganti missing value
Mode <- function(x) { 
  ux <- unique(x[!is.na(x)])
  ux[which.max(tabulate(match(x, ux)))]
}

# Handling Missing Value dengan fungsi mode
df <- df %>%
  mutate(across(where(is.character) | where(is.factor),
                ~ ifelse(is.na(.), Mode(.), .)))

# ============================================================
# 4. PRE-PROCESSING
# ============================================================
#
df_original_backup <- df

# Filter Tabel Skewness
df <- df %>% mutate(across(where(is.numeric), ~ log(. + 1)))

df_cluster <- df %>% select(where(is.numeric))

# Normalization
preproc_cluster <- preProcess(df_cluster, method = c("center", "scale"))
df_cluster_scaled <- predict(preproc_cluster, df_cluster)

# ============================================================
# 5. PROCESSING
# ============================================================

# ============================================================
# CLUSTERING (K-Means)
# ============================================================

wss <- sapply(1:10, function(k)
  kmeans(df_cluster_scaled, k, nstart = 10)$tot.withinss)

plot(1:10, wss, type = "b",
     main = "Elbow Method",
     xlab = "k", ylab = "Within Sum of Squares")

set.seed(123)
k <- 2
kmeans_model <- kmeans(df_cluster_scaled, centers = k, nstart = 25)

fviz_cluster(kmeans_model, df_cluster_scaled,
             main = "Cluster Visualization")

df$cluster <- as.factor(kmeans_model$cluster)

cat("\n=== INTERPRETASI CLUSTER ===\n")
cluster_summary <- df_original_backup %>%
  mutate(cluster = df$cluster) %>%
  group_by(cluster) %>%
  summarise(
    count = n(),
    mean_exam_score = mean(Exam_Score),
    mean_hours_studied = mean(Hours_Studied),
    mean_attendance = mean(Attendance),
    mean_sleep_hours = mean(Sleep_Hours)
  )
print(cluster_summary)

cluster_with_highest_score <- cluster_summary %>%
  filter(mean_exam_score == max(mean_exam_score)) %>%
  pull(cluster)

cat("\nCluster dengan Exam Score tertinggi:", as.character(cluster_with_highest_score), "\n")
cat("Cluster ini akan dilabel sebagai 'pass'\n")

df$Exam_score_pass <- as.factor(ifelse(df$cluster == cluster_with_highest_score, "pass", "fail"))

cat("\nDistribusi kelas berdasarkan cluster:\n")
print(table(df$Exam_score_pass))

# ============================================================
# CLASSIFICATION (TARGET: Exam_score_pass)
# ============================================================

target_col <- "Exam_score_pass"

df_encoded <- df %>%
  mutate(across(
    where(is.character) & !all_of(target_col),
    ~ as.numeric(factor(.))
  )) %>%
  mutate(across(
    where(is.factor) & !all_of(target_col),
    ~ as.numeric(as.character(.))
  ))

df_encoded[[target_col]] <- df$Exam_score_pass

df_encoded <- df_encoded %>% select(-any_of(c("Exam_Score", "cluster")))

preproc_class <- preProcess(df_encoded %>% select(-all_of(target_col)),
                            method = c("center", "scale"))

df_scaled <- df_encoded
df_scaled[, setdiff(names(df_scaled), target_col)] <-
  predict(preproc_class, df_encoded %>% select(-all_of(target_col)))

set.seed(42)
idx <- createDataPartition(df_scaled[[target_col]], p = 0.8, list = FALSE)
train <- df_scaled[idx, ]
test  <- df_scaled[-idx, ]

str(train[[target_col]])

model_dt <- rpart(as.formula(paste(target_col, "~ .")),
                  data = train, method = "class")
pred_dt <- predict(model_dt, test, type = "class")
cm_dt <- confusionMatrix(pred_dt, test[[target_col]])

model_rf <- randomForest(as.formula(paste(target_col, "~ .")), data = train)
pred_rf <- predict(model_rf, test)
cm_rf <- confusionMatrix(pred_rf, test[[target_col]])

model_svm <- svm(as.formula(paste(target_col, "~ .")), data = train)
pred_svm <- predict(model_svm, test)
cm_svm <- confusionMatrix(pred_svm, test[[target_col]])

# ============================================================
# 6. EVALUATION SUMMARY
# ============================================================
cat("Decision Tree Accuracy:", cm_dt$overall["Accuracy"], "\n")
cat("Random Forest Accuracy:", cm_rf$overall["Accuracy"], "\n")
cat("SVM Accuracy:", cm_svm$overall["Accuracy"], "\n")

# ============================================================
# 7. ASSOCIATION RULES (Apriori)
# ============================================================

df_factor <- df %>%
  mutate_if(is.numeric, function(x) as.factor(ntile(x, 5)))

trans <- as(df_factor, "transactions")

rules <- apriori(trans,
                 parameter = list(supp = 0.05, conf = 0.4))

inspect(head(sort(rules, by = "lift"), 10))
plot(rules, method = "graph", engine = "htmlwidget")

# ============================================================
# DONE
# ============================================================