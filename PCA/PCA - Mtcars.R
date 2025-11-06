# Load dataset mtcars
data(mtcars)

# Cek strukturnya
str(mtcars)

# Cek missing values
any(is.na(mtcars))

# Scaling dataset mtcars
centered_data <- scale(mtcars, center = TRUE, scale = TRUE)

# Hitung matrix covariance
cov_matrix <- cov(centered_data)

# Melakukan dekomposisi eigen pada matriks kovariansi
eigen_result <- eigen(cov_matrix)

# Simpan nilai eigen
eigenvalues <- eigen_decomp$values

# Urutkan eigenvector berdasarkan eigenvalue 
eigen_result$vectors <- eigen_result$vectors[, order(eigen_result$values, 
                                                     decreasing = TRUE)]

# Transformasi ke ruang komponen utama
transformed_data <- centered_data %*% eigen_result$vectors

# Ambil 2 komponen utama pertama
k <- 2 
transformed_data_k <- transformed_data[, 1:k]
colnames(transformed_data_k) <- c("PC1", "PC2")

# Lihat hasil
head(transformed_data_k)

# Visualisasi hasil PCA
plot(
  transformed_data_k,
  main = "Visualisasi PCA mtcars",
  xlab = "Komponen Utama 1 (PC1)",
  ylab = "Komponen Utama 2 (PC2)",
  pch = 19,            
  col = "steelblue"
)

