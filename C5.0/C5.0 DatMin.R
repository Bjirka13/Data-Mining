# Load library
library(C50)
library(readr)
library(rpart)
library(rpart.plot)
library(caret)
library(ggplot2)
library(FSelector)

# ============= IMPORT DATASET =============
mushroom <- read_csv("mushrooms.csv", show_col_types = FALSE)

# ============= EDA =============
cat("\n=== Struktur Dataset ===\n")
str(mushroom)
cat("\nMissing value:", sum(is.na(mushroom)), "\n")
cat("Data duplikat:", sum(duplicated(mushroom)), "\n")

cat("\nJumlah data per kelas target:\n")
print(table(mushroom$class))

# Grafik proporsi kelas
target_df <- as.data.frame(table(mushroom$class))
colnames(target_df) <- c("class", "count")
target_df$percent <- round(target_df$count / sum(target_df$count) * 100, 2)

ggplot(target_df, aes(x = "", y = percent, fill = factor(class))) +
  geom_col(width = 1, color = "white") +
  coord_polar(theta = "y") +
  geom_text(aes(label = paste0(percent, "%")),
            position = position_stack(vjust = 0.5), size = 4) +
  scale_fill_manual(values = c("#1f77b4", "#ff7f0e")) +
  labs(
    title = "Proporsi Kelas Jamur (Edible vs Poisonous)",
    fill = "Kelas (e = Edible, p = Poisonous)"
  ) +
  theme_void(base_size = 13)

cat("\nDataset ini berisi fitur kategorikal, sehingga deteksi outlier tidak relevan.\n")

# ============= 3. PRE-PROCESSING =============
# Konversi semua kolom menjadi numerik
mushroom[] <- lapply(mushroom, as.factor)

# ============= 4. ANALISIS RELEVANSI FITUR =============
cat("\n=== Analisis Relevansi Fitur terhadap Target (class) ===\n")

ig <- information.gain(class ~ ., mushroom)
ig <- ig[order(-ig$attr_importance), , drop = FALSE]

cat("\nUrutan kepentingan fitur berdasarkan Information Gain:\n")
print(ig)

# Pilih fitur dengan relevansi kuat (misal IG > 0.1)
strong_features <- rownames(ig[ig$attr_importance > 0.1, , drop = FALSE])
cat("\nFitur relevan (IG > 0.1):\n")
print(strong_features)

# Dataset hanya dengan fitur relevan
mushroom_relevant <- mushroom[, c("class", strong_features)]

# ============= 5. SPLIT MANUAL (TRAIN-TEST) =============
set.seed(123)
train_index <- sample(1:nrow(mushroom_relevant), 0.7 * nrow(mushroom_relevant))
train_data <- mushroom_relevant[train_index, ]
test_data  <- mushroom_relevant[-train_index, ]

cat("\nJumlah data training:", nrow(train_data), "\n")
cat("Jumlah data testing :", nrow(test_data), "\n")

# ============= 6. MODEL C5.0 =============
model_c50 <- C5.0(class ~ ., data = train_data)
cat("\n=== Ringkasan Model C5.0 ===\n")
summary(model_c50)

# ============= 7. EVALUASI MODEL =============
preds <- predict(model_c50, test_data)
cm <- confusionMatrix(preds, test_data$class)

cat("\n=== Confusion Matrix ===\n")
print(cm$table)
cat("\nAkurasi =", round(cm$overall["Accuracy"] * 100, 2), "%\n")
cat("Kappa =", round(cm$overall["Kappa"], 3), "\n")

# ============= 8. PIE CHART PROPORSI LABEL DI DATA TESTING =============
cat("\n=== Distribusi Label pada Data Testing ===\n")

# Hitung proporsi kelas aktual di data testing
test_dist <- as.data.frame(table(test_data$class))
colnames(test_dist) <- c("class", "count")
test_dist$percent <- round(test_dist$count / sum(test_dist$count) * 100, 2)

print(test_dist)

# Buat grafik pie chart
ggplot(test_dist, aes(x = "", y = percent, fill = factor(class))) +
  geom_col(width = 1, color = "white") +
  coord_polar(theta = "y") +
  geom_text(aes(label = paste0(percent, "%")),
            position = position_stack(vjust = 0.5), size = 4) +
  scale_fill_manual(values = c("#1f77b4", "#ff7f0e")) +
  labs(
    title = "Proporsi Kelas pada Data Testing (Balance Check)",
    fill = "Kelas (e = Edible, p = Poisonous)"
  ) +
  theme_void(base_size = 13)


# ============= 9. PENTINGNYA VARIABEL =============
cat("\n=== Pentingnya Variabel ===\n")
importance <- C5imp(model_c50, metric = "usage", pct = TRUE)
print(importance)

barplot(
  importance[, 1],
  names.arg = rownames(importance),
  las = 2,
  col = "skyblue",
  main = "Variable Importance (Mushroom Edibility)",
  ylab = "Pengaruh (%)"
)

# ============= 10. VISUALISASI POHON KEPUTUSAN =============
rpart_model <- rpart(class ~ ., data = train_data, method = "class")
rpart.plot(
  rpart_model,
  main = "Pohon Keputusan Prediksi Kelas Jamur (Fitur Relevan)",
  extra = 104,
  box.palette = "GnBu",
  shadow.col = "gray",
  nn = TRUE
)

# ============= 11. SIMPAN MODEL =============
saveRDS(model_c50, "model_c50_mushroom_manualsplit.rds")
cat("\nModel berhasil disimpan ke file: model_c50_mushroom_manualsplit.rds\n")
