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
library(pROC)

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

# Deklarasi Fungsi Mode untuk imputasi missing value
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

# Log Transform hanya untuk Kolom Skewness
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

cat("Visualisasi Elbow Method")
wss <- sapply(1:10, function(k)
  kmeans(df_cluster_scaled, k, nstart = 10)$tot.withinss)

plot(1:10, wss, type = "b",
     main = "Elbow Method",
     xlab = "k", ylab = "Within Sum of Squares")

# Deklarasi Jumlah Cluster = 2
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
cat("========== Preprocessing Tambahan ==========")
# Deklarasi Target Prediksi
target_col <- "Exam_score_pass"

# Encoding Data Kategorik (LabelEncoding)
df_encoded <- df %>%
  mutate(across(
    where(is.character) & !all_of(target_col),
    ~ as.numeric(factor(.))
  )) %>%
  mutate(across(
    where(is.factor) & !all_of(target_col),
    ~ as.numeric(as.character(.))
  ))

# Tambahkan Target prediksi ke dalam Dataset
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

# ============================================================
# DECISION TREE
# ============================================================
model_dt <- rpart(as.formula(paste(target_col, "~ .")),
                  data = train, method = "class")

pred_dt_train <- predict(model_dt, train, type = "class")
pred_dt_test  <- predict(model_dt, test,  type = "class")

cm_dt_train <- confusionMatrix(pred_dt_train, train[[target_col]])
cm_dt_test  <- confusionMatrix(pred_dt_test,  test[[target_col]])

# ============================================================
# RANDOM FOREST
# ============================================================
model_rf <- randomForest(as.formula(paste(target_col, "~ .")), data = train)

pred_rf_train <- predict(model_rf, train)
pred_rf_test  <- predict(model_rf, test)

cm_rf_train <- confusionMatrix(pred_rf_train, train[[target_col]])
cm_rf_test  <- confusionMatrix(pred_rf_test,  test[[target_col]])

# ============================================================
# SVM
# ============================================================
model_svm <- svm(as.formula(paste(target_col, "~ .")), data = train)

pred_svm_train <- predict(model_svm, train)
pred_svm_test  <- predict(model_svm, test)

cm_svm_train <- confusionMatrix(pred_svm_train, train[[target_col]])
cm_svm_test  <- confusionMatrix(pred_svm_test,  test[[target_col]])

# ============================================================
# 6. EVALUATION SUMMARY (COMPACT VERSION)
# ============================================================

# --- 1. Fungsi Persiapan Data (Sama seperti sebelumnya) ---
prepare_cm_data <- function(cm, model_name) {
  df_cm <- as.data.frame(cm$table)
  accuracy <- round(cm$overall['Accuracy'], 4)
  df_cm$Model_Label <- paste0(model_name, "\n(Acc: ", accuracy, ")") # Singkat teks "Accuracy" jadi "Acc"
  return(df_cm)
}

# --- 2. Menggabungkan Data ---
data_dt  <- prepare_cm_data(cm_dt_test,  "Decision Tree")
data_rf  <- prepare_cm_data(cm_rf_test,  "Random Forest")
data_svm <- prepare_cm_data(cm_svm_test, "SVM")

all_cm_data <- rbind(data_dt, data_rf, data_svm)

# --- 3. Plotting Compact ---
ggplot(all_cm_data, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  
  # UBAH 1: Perkecil ukuran angka di dalam kotak (misal jadi 4)
  geom_text(aes(label = Freq), size = 4, color = "black") +
  
  facet_wrap(~ Model_Label, ncol = 3) + 
  scale_fill_gradient(low = "#e1f5fe", high = "#0277bd") +
  
  labs(title = "Model Comparison (Test Set)",
       x = "Actual",   # Label sumbu dipersingkat
       y = "Predicted") +
  
  # UBAH 2: Perkecil base_size font (misal jadi 11)
  theme_minimal(base_size = 11) +
  
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    strip.text = element_text(face = "bold", size = 10),
    panel.grid = element_blank(),
    legend.position = "right",
    legend.title = element_text(size = 10), 
    legend.text = element_text(size = 9)
  ) +
  
  coord_fixed()

# ============================================================
# 6.5. VISUALISASI TAMBAHAN (ROC CURVE & VARIABLE IMPORTANCE)
# ============================================================
 
# Khusus SVM perlu di-train ulang dengan probability = TRUE
model_svm_prob <- svm(as.formula(paste(target_col, "~ .")), 
                      data = train, 
                      probability = TRUE) # PENTING!

# 1. Decision Tree
pred_dt_prob <- predict(model_dt, test, type = "prob")[, "pass"]

# 2. Random Forest
pred_rf_prob <- predict(model_rf, test, type = "prob")[, "pass"]

# 3. SVM 
pred_svm_attr <- predict(model_svm_prob, test, probability = TRUE)
pred_svm_prob <- attr(pred_svm_attr, "probabilities")[, "pass"]

roc_dt  <- roc(test[[target_col]], pred_dt_prob, levels = c("fail", "pass"))
roc_rf  <- roc(test[[target_col]], pred_rf_prob, levels = c("fail", "pass"))
roc_svm <- roc(test[[target_col]], pred_svm_prob, levels = c("fail", "pass"))

roc_list <- list(
  "Decision Tree" = roc_dt,
  "Random Forest" = roc_rf,
  "SVM"           = roc_svm
)

# Hitung AUC untuk ditampilkan di Legenda
auc_values <- lapply(roc_list, function(x) round(auc(x), 4))
legend_labels <- paste(names(roc_list), "- AUC:", auc_values)

cat("Plot ROC")
ggroc(roc_list, size = 1) +
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), 
               color = "grey", linetype = "dashed") +
  scale_color_manual(values = c("#FF6B6B", "#4ECDC4", "#1A535C"), 
                     labels = legend_labels) +
  labs(title = "Model Comparison: ROC Curves",
       subtitle = "Semakin kurva mendekati pojok kiri atas, semakin baik modelnya",
       color = "Model & AUC Score") +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, size = 10, color = "grey40"),
    legend.position = "bottom",
    legend.box.background = element_rect(color = "grey90"),
    panel.grid.minor = element_blank()
  )

cat("VARIABLE IMPORTANCE")
# Ini membantu menjelaskan "Kenapa siswa lulus/gagal?"
importance_rf <- varImp(model_rf)
importance_df <- data.frame(Feature = rownames(importance_rf), 
                            Importance = importance_rf$Overall)

ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() + # Memutar bar chart agar label terbaca
  labs(title = "Faktor Paling Berpengaruh Terhadap Kelulusan",
       subtitle = "Variable Importance (Random Forest)",
       x = "Feature", y = "Importance Score") +
  theme_minimal()

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