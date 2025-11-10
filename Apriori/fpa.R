install.packages("arules")
install.packages("arulesViz")

library(arules)
library(arulesViz)
library(dplyr)
library(ggplot2)

# Load Dataset
groceries_raw <- read.csv("Groceries_dataset.csv")

## EDA
# Struktur Dataset
cat("Struktur data:\n")
print(str(groceries_raw))
cat("\n")

# Melihat 6 data teratas
head(groceries_raw)

# Ketahui Missing Values
sum(is.na(groceries_raw))

colnames(groceries_raw) <- c("Member_number", "Date", "itemDescription")

## Preprocessing

groceries_raw %>%
  group_by(Member_number, Date, itemDescription) %>%
  filter(n() > 1)

# Menghapus duplikat
groceries_raw <- groceries_raw %>%
  distinct()

groceries_grouped <- groceries_raw %>%
  group_by(Member_number, Date) %>%
  summarise(items = paste(unique(itemDescription), collapse = ","), .groups = "drop")

write.csv(groceries_grouped["items"], 
          "Groceries_transactions.csv", 
          row.names = FALSE, 
          quote = FALSE)

cat("File transaksi bersih tersimpan: Groceries_transactions.csv\n\n")

data_groceries <- read.transactions(
  "Groceries_transactions.csv",
  sep = ",",
  format = "basket",
  rm.duplicates = TRUE
)

cat("Ringkasan Data:\n")
summary(data_groceries)

itemFrequencyPlot(
  data_groceries,
  topN = 20,
  type = "absolute",
  col = "skyblue",
  main = "20 Produk Paling Sering Dibeli",
  xlab = "Produk",
  ylab = "Frekuensi"
)

## Penerapan Apriori
rules_groceries <- apriori(
  data_groceries,
  parameter = list(supp = 0.001, conf = 0.2, minlen = 2)
)

cat("\n Jumlah aturan yang ditemukan:", length(rules_groceries), "\n\n")

if (length(rules_groceries) > 0) {
  top_rules <- sort(rules_groceries, by = "lift", decreasing = TRUE)
  cat("10 Aturan dengan Lift Tertinggi:\n")
  inspect(head(top_rules, 10))
} else {
  cat("Tidak ada aturan ditemukan. Coba turunkan nilai support/confidence.\n")
}

if (length(top_rules) > 0) {
  n_rules <- min(30, length(top_rules))  # ambil maksimal 30, tapi tidak lebih dari jumlah rule yang ada
  plot(top_rules[1:n_rules], method = "graph", engine = "igraph")
  plot(top_rules[1:n_rules], method = "paracoord", control = list(reorder = TRUE))
}

if (length(rules_groceries) > 0) {
  metrics <- quality(rules_groceries)
  
  hist(metrics$lift,
       main = "Distribusi Nilai Lift",
       xlab = "Lift",
       col = "orange",
       breaks = 30)
  
  hist(metrics$confidence,
       main = "Distribusi Nilai Confidence",
       xlab = "Confidence",
       col = "lightgreen",
       breaks = 30)
  
  hist(metrics$support,
       main = "Distribusi Nilai Support",
       xlab = "Support",
       col = "skyblue",
       breaks = 30)
}

if (length(rules_groceries) > 0) {
  metrics <- quality(rules_groceries)[1:50, ]
  
  ggplot(metrics, aes(x = 1:50)) +
    geom_line(aes(y = support, color = "Support"), linewidth = 1.2) +
    geom_line(aes(y = confidence, color = "Confidence"), linewidth = 1.2) +
    geom_line(aes(y = lift / 10, color = "Lift (dibagi 10)"),
              linewidth = 1.2, linetype = "dashed") +
    labs(title = "Tren Nilai Support, Confidence, dan Lift pada 50 Rule Teratas",
         x = "Rule ke-",
         y = "Nilai",
         color = "Metrik") +
    theme_minimal()
}

if (length(rules_groceries) > 0) {
  df <- as.data.frame(quality(rules_groceries))
  
  ggplot(df[1:20, ], aes(x = reorder(as.factor(1:20), -lift))) +
    geom_col(aes(y = lift), fill = "tomato", alpha = 0.7) +
    geom_line(aes(y = confidence * 5, group = 1), color = "blue", linewidth = 1) +
    geom_point(aes(y = confidence * 5), color = "blue") +
    labs(title = "Lift dan Confidence dari 20 Rule Teratas",
         x = "Rule ke-",
         y = "Lift (dan Confidence x5)") +
    theme_minimal()
}

cat("\n✅ Analisis Frequent Pattern Analysis (Apriori) selesai\n")