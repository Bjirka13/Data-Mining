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
glimpse(df)
summary(df)

# Total missing value di seluruh dataset
print(sum(is.na(df)))

# Jumlah missing value per kolom
print(colSums(is.na(df)))

# Jumlah baris duplikat
print(sum(duplicated(df)))


# Cek Distribusi Fitur Numerik
cat("HISTOGRAM DISTRIBUSI SETIAP FITUR NUMERIK")
numeric_df <- df %>% select(where(is.numeric))

# long format untuk facet
numeric_long <- melt(numeric_df)

ggplot(numeric_long, aes(x = value)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white", alpha = 0.8) +
  facet_wrap(~ variable, scales = "free", ncol = 3) +
  theme_minimal(base_size = 14) +
  labs(
    title = "Distribusi Histogram Fitur Numerik",
    x = "Value",
    y = "Frequency"
  ) +
  theme(panel.grid = element_blank())


# Deteksi Outlier
cat("OUTLIER DETECTION (IQR METHOD)")

numeric_df <- df %>% select(where(is.numeric))

detect_outliers <- function(x) {
  Q1  <- quantile(x, 0.25, na.rm = TRUE)
  Q3  <- quantile(x, 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  sum(x < lower_bound | x > upper_bound)
}

outlier_counts <- sapply(numeric_df, detect_outliers)

print(outlier_counts)


cat("CORRELATION HEATMAP")
corr_matrix <- cor(numeric_df, use = "complete.obs")
corr_long <- melt(corr_matrix, varnames = c("Var1", "Var2"), value.name = "Correlation")

ggplot(corr_long, aes(Var1, Var2, fill = Correlation)) +
  geom_tile(color = "white") +
  geom_text(aes(label = round(Correlation, 2)), size = 5) +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white",
                       midpoint = 0, limit = c(-1, 1)) +
  theme_minimal(base_size = 14) +
  labs(
    title = "Correlation Matrix (ggplot2)",
    x = "",
    y = ""
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid = element_blank()
  )

# ============================================================
# 3. DATA CLEANING
# ============================================================

# Hapus Noise
df <- df %>% filter(Exam_Score <= 100)

# Hapus duplikat baris jika ada
df <- df %>% distinct()



## Normalization
preproc <- preProcess(df, method = c("center", "scale"))
df_scaled <- predict(preproc, df)

# ============================================================
# 4. CLUSTERING (K-Means)
# ============================================================
df_cluster <- df_scaled %>% select(where(is.numeric))

# Elbow method
wss <- sapply(1:10, function(k) kmeans(df_cluster, k, nstart=10)$tot.withinss)
plot(1:10, wss, type="b", main="Elbow Method")

# Pilih k = 3
set.seed(123)
k <- 3
kmeans_model <- kmeans(df_cluster, centers=k, nstart=25)

# Visualisasi cluster
fviz_cluster(kmeans_model, df_cluster, main="Cluster Visualization")

# Tambah cluster ke dataset asli
df$cluster <- as.factor(kmeans_model$cluster)

# ============================================================
# 5. CLASSIFICATION (TARGET: Exam_score_pass)
# ============================================================
target_col <- "Exam_score_pass"

df_class <- df_scaled
df_class[[target_col]] <- as.factor(df[[target_col]])

set.seed(42)
idx <- createDataPartition(df_class[[target_col]], p=0.8, list=FALSE)
train <- df_class[idx, ]
test  <- df_class[-idx, ]

# ------------------------------------------------------------
# (A) Decision Tree
# ------------------------------------------------------------
model_dt <- rpart(as.formula(paste(target_col, "~ .")), data=train, method="class")
pred_dt  <- predict(model_dt, test, type="class")
cm_dt <- confusionMatrix(pred_dt, test[[target_col]])

# ------------------------------------------------------------
# (B) Random Forest
# ------------------------------------------------------------
model_rf <- randomForest(as.formula(paste(target_col, "~ .")), data=train)
pred_rf  <- predict(model_rf, test)
cm_rf <- confusionMatrix(pred_rf, test[[target_col]])

# ------------------------------------------------------------
# (C) SVM
# ------------------------------------------------------------
model_svm <- svm(as.formula(paste(target_col, "~ .")), data=train)
pred_svm  <- predict(model_svm, test)
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

# Numeric → binned factor
df_factor <- df %>% mutate_if(is.numeric, function(x) as.factor(ntile(x, 5)))

trans <- as(df_factor, "transactions")

rules <- apriori(trans,
                 parameter = list(supp=0.05, conf=0.4))

inspect(head(sort(rules, by="lift"), 10))
plot(rules, method="graph", engine="htmlwidget")

# ============================================================
# DONE
# ============================================================
