install.packages(c("palmerpenguins", "e1071", "caTools", "dplyr",
                    "ggplot2", "caret", "klaR", "FSelector"))

library(palmerpenguins)
library(caTools)
library(dplyr)
library(ggplot2)
library(caret)
library(FSelector)

# ========== EDA ==========
data("penguins")
str(penguins)
summary(penguins)

# cek jumlah missing value
sum(is.na(data_penguins))

# cek jumlah duplikasi data
sum(duplicated(data_penguins))

# ========== Cleaning & Processing Data ==========
penguins <- na.omit(penguins)

data_penguins <- penguins %>%
  dplyr::select(species, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g)


# Standarisasi fitur numerik
num_cols <- sapply(data_penguins, is.numeric)
data_penguins[num_cols] <- scale(data_penguins[num_cols])

# ========== Deteksi Outlier ==========
cat("\n=== Deteksi Outlier (menggunakan Z-score) ===\n")

# Menghitung Z-score
z_scores <- as.data.frame(scale(data_penguins[, num_cols]))

# Menentukan ambang batas outlier (umumnya |Z| > 3)
outlier_flags <- apply(z_scores, 2, function(x) abs(x) > 3)

# Hitung jumlah outlier per fitur
outlier_count <- colSums(outlier_flags)
print(outlier_count)

# Jumlah baris yang mengandung outlier
total_outlier_rows <- sum(apply(outlier_flags, 1, any))
cat("Jumlah baris yang mengandung outlier:", total_outlier_rows, "\n")


# ========== Analisis Relevansi Fitur ==========
cat("\nInformasi Gain antar Fitur dan Target:\n")
ig <- information.gain(species ~ ., data_penguins)
ig <- ig[order(-ig$attr_importance), , drop = FALSE]
cat("\nUrutan kepentingan fitur berdasarkan Information Gain:\n")
print(ig)

# ========== Train-Test Split dengan Cross Validation ==========
set.seed(123)
train_control <- trainControl(method = "cv", number = 10)

model_nb <- train(
  species ~ .,
  data = data_penguins,
  method = "naive_bayes",
  trControl = train_control,
  tuneGrid = expand.grid(
    laplace = 1,
    usekernel = TRUE,
    adjust = 1
  )
)


# ========== Prediksi ==========
print(model_nb)
prediksi <- predict(model_nb, data_penguins)

cat("\n==================== Output Model ====================\n")
predict(model_nb, data_penguins, type = "raw")  # menghasilkan probabilitas per kelas
predict(model_nb, data_penguins, type = "prob") # menghasilkan label prediksi (kelas)
cat("========================================================\n")



# ========== Cek Kinerja Model ==========

# Pisahkan data latih & uji ulang (karena sebelumnya pakai cross-validation)
set.seed(123)
split <- sample.split(data_penguins$species, SplitRatio = 0.8)
train_data <- subset(data_penguins, split == TRUE)
test_data  <- subset(data_penguins, split == FALSE)

# Latih ulang model pada data latih
model_nb_split <- train(
  species ~ .,
  data = train_data,
  method = "naive_bayes",
  trControl = trainControl(method = "none"), # tanpa CV karena kita pakai hold-out
  tuneGrid = expand.grid(
    laplace = 1,
    usekernel = TRUE,
    adjust = 1
  )
)

# Prediksi pada data latih dan data uji
pred_train <- predict(model_nb_split, train_data)
pred_test  <- predict(model_nb_split, test_data)

# Hitung akurasi keduanya
acc_train <- mean(pred_train == train_data$species)
acc_test  <- mean(pred_test == test_data$species)

cat("\n==================== CEK OVERFITTING / UNDERFITTING ====================\n")
cat("Akurasi Data Latih :", round(acc_train * 100, 2), "%\n")
cat("Akurasi Data Uji   :", round(acc_test * 100, 2), "%\n")

# Interpretasi otomatis
selisih <- abs(acc_train - acc_test)

if (acc_train > acc_test && selisih > 0.1) {
  cat("⚠Model kemungkinan OVERFITTING (terlalu bagus di data latih, lemah di data uji)\n")
} else if (acc_train < 0.7 && acc_test < 0.7) {
  cat("⚠Model kemungkinan UNDERFITTING (belum mampu menangkap pola data)\n")
} else {
  cat("odel tampak SEIMBANG (tidak overfit maupun underfit)\n")
}

# ========== Evaluasi Model ==========

cat("\n=== CONFUSION MATRIX ===\n")
conf_mat <- confusionMatrix(prediksi, data_penguins$species)
print(conf_mat)

akurasi_total <- round(conf_mat$overall["Accuracy"] * 100, 2)
total_data <- sum(conf_mat$table)
benar <- sum(diag(conf_mat$table))
salah <- total_data - benar
persentase_benar <- round((benar / total_data) * 100, 2)
persentase_salah <- round((salah / total_data) * 100, 2)

cat("\n==================== AKURASI Evaluasi ====================\n")
cat("Akurasi Total Model : ", akurasi_total, "%\n")
cat("Jumlah Data Benar   : ", benar, "(", persentase_benar, "% )\n")
cat("Jumlah Data Salah   : ", salah, "(", persentase_salah, "% )\n")
cat("========================================================\n")


# Tambahan: tampilkan F1-score per kelas
cat("F1-score per kelas:\n")
print(round(conf_mat$byClass[, "F1"], 3))

# ========== Analisis Persentase Bias Model ==========
hasil <- data.frame(
  Aktual = data_penguins$species,
  Prediksi = prediksi
)
kesalahan <- hasil %>% filter(Aktual != Prediksi)

cat("\n=== Contoh 10 Data yang Salah ===\n")
print(head(kesalahan, 10))

# ========== Visualisasi Korelasi antar Fitur dengan Relasi Paling Kuat ==========
ggplot(data_penguins, aes(x = flipper_length_mm, y = body_mass_g, color = species)) +
  geom_point(size = 3, alpha = 0.8) +
  labs(
    title = "Visualisasi Pemisahan Spesies Penguin",
    subtitle = "Menggunakan Fitur Flipper Length vs Body Mass",
    x = "Flipper Length (standar)",
    y = "Body Mass (standar)"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold"))

cm_df <- as.data.frame(conf_mat$table)
ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 5) +
  scale_fill_gradient(low = "skyblue", high = "darkblue") +
  labs(title = "Confusion Matrix Naive Bayes (CV 10-Fold)") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold"))

# ========== Kesimpulan ==========
cat("\n==================== KESIMPULAN ====================\n")
cat("Model Naive Bayes dengan kernel density dan cross-validation\n")
cat("menunjukkan performa klasifikasi yang stabil dan interpretatif.\n")
cat("Fitur paling berpengaruh biasanya: flipper_length_mm dan body_mass_g.\n")
cat("====================================================\n")

