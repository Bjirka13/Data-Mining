# Load dataset iris
data(penguins)

# Ambil kolom numerik
iris_data <- penguins[, 1:4]

# Scaling dat# Scaling dat# Scaling data (rata-rata = 0, variansi = 1)
standardized_data <- scale(iris_data)

# Hitung matrix covariance
cov_matrix <- cov(standardized_data)

# Melakukan dekomposisi eigen pada matriks kovariansi
eigen_decomp <- eigen(cov_matrix)

# Simpan nilai eigen
eigenvalues <- eigen_decomp$values

# Simpan vektor eigen
eigenvectors <- eigen_decomp$vectors

# Hitung komponen utama
principal_components <- standardized_data %*% eigenvectors

# Gabungkan komponen utama dengan kolom species untuk interpretasi
pca_result <- cbind(principal_components, Species = iris$Species)

# Print hasil PCA
print(pca_result)


# Visualisasi hasil PCA
library(ggplot2)

ggplot(pca_df, aes(x = PC1, y = PC2, color = Species)) +
  geom_point(size = 3) +
  theme_minimal() +
  labs(title = "Visualisasi PCA", x = "Principal Component 1", y = "Principal Component 2")

