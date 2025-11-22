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


# ============================================================
# 2. LOAD DATA
# ============================================================
df <- read_csv("StudentPerformanceFactors.csv")

# Struktur Dataset
glimpse(df)

# Ringkasan Deskriptif Dataset
summary(df)

# Cek Missing Value
print(colSums(is.na(df)))

# Cek Duplikat Value
print(sum(duplicated(df)))


# ANALISIS KORELASI FITUR (NUMERIK)
numeric_df <- df %>% select(where(is.numeric))

# Hitung korelasi
corr_matrix <- cor(numeric_df, use = "complete.obs")

print(corr_matrix)

# ============================================================
# CORRELATION HEATMAP (PURE GGPLOT2)
# ============================================================

library(reshape2)

# Ambil fitur numerik
numeric_df <- df %>% select(where(is.numeric))

# Hitung matriks korelasi
corr_matrix <- cor(numeric_df, use = "complete.obs")

# Ubah jadi long format
corr_long <- melt(corr_matrix, varnames = c("Var1", "Var2"), value.name = "Correlation")

# Plot korelasi dengan ggplot2
ggplot(corr_long, aes(Var1, Var2, fill = Correlation)) +
  geom_tile(color = "white") +
  geom_text(aes(label = round(Correlation, 2)), size = 5) +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white",
                       midpoint = 0, limit = c(-1, 1)) +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid = element_blank()
  ) +
  labs(
    title = "Correlation Matrix (ggplot2)",
    x = "",
    y = ""
  )


# ============================================================
# 3. DATA CLEANING
# ============================================================

## Missing values
df <- df %>% drop_na()

## Convert character → factor
df <- df %>% mutate_if(is.character, as.factor)

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
