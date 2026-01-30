#!/usr/bin/env Rscript
# ============================================================
# MMUPHin batch-effect correction pipeline
# - Automatic detection of normalization status
# - Removes batch effects while preserving phenotype labels
# ============================================================

suppressPackageStartupMessages({
  library(MMUPHin)
})

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
message("Loading data...")

abd <- read.csv(
  "nonchina_A50_full.csv",
  row.names = 1,
  check.names = FALSE
)  # rows = samples, columns = taxa

meta_raw <- read.csv(
  "nonchina_A50_meta.csv",
  check.names = FALSE
)

# Standardize metadata column names
colnames(meta_raw)[1:3] <- c("Sample", "group", "label")
meta_raw <- unique(meta_raw)
rownames(meta_raw) <- meta_raw$Sample

# ------------------------------------------------------------
# Align samples
# ------------------------------------------------------------
common_samples <- intersect(rownames(abd), rownames(meta_raw))
if (length(common_samples) < 2L) {
  stop("Too few overlapping samples. Please check sample identifiers.")
}

abd  <- abd[common_samples, , drop = FALSE]
meta <- meta_raw[common_samples, , drop = FALSE]

# ------------------------------------------------------------
# Basic validation
# ------------------------------------------------------------
if (!all(sapply(abd, is.numeric))) {
  stop("Abundance table contains non-numeric columns. Retain taxa abundances only.")
}

if (any(is.na(as.matrix(abd)))) {
  warning("NA values detected and will be set to 0.")
  abd[is.na(abd)] <- 0
}

if (any(abd < 0)) {
  stop("Negative values detected. Input should be non-negative counts or relative abundances.")
}

# ------------------------------------------------------------
# Check normalization status (row sum ≈ 1)
# ------------------------------------------------------------
row_sums <- rowSums(abd)
normalized_flag <- all(abs(row_sums - 1) < 1e-6)

if (!normalized_flag) {
  message("Input not normalized. Applying total-sum scaling (TSS).")

  zero_rows <- which(row_sums == 0)
  if (length(zero_rows) > 0) {
    warning(sprintf(
      "%d samples with zero total abundance removed: %s",
      length(zero_rows),
      paste(rownames(abd)[zero_rows], collapse = ", ")
    ))
    abd <- abd[-zero_rows, , drop = FALSE]
    meta <- meta[rownames(abd), , drop = FALSE]
    row_sums <- rowSums(abd)
  }

  abd <- sweep(abd, 1, row_sums, "/")

} else {
  message("Input already appears to be relative abundance. Skipping normalization.")
}

# Ensure numerical stability
abd[!is.finite(as.matrix(abd))] <- 0

# ------------------------------------------------------------
# Prepare input for MMUPHin (features × samples)
# ------------------------------------------------------------
feat_by_samp <- t(as.matrix(abd))   # rows = taxa, columns = samples
meta$group <- as.factor(meta$group)
meta$label <- as.factor(meta$label)

# ------------------------------------------------------------
# Run MMUPHin batch correction
# ------------------------------------------------------------
message("Running MMUPHin::adjust_batch...")

adj <- adjust_batch(
  feature_abd = feat_by_samp,
  batch       = "group",   # batch variable to remove
  covariates  = "label",   # biological signal to preserve
  data        = meta,
  control     = list(
    zero_inflation  = TRUE,
    diagnostic_plot = "nonchina_A50_MMUPHin_diagnostic.pdf",
    verbose         = TRUE
  )
)

adj_feat_by_samp <- adj$feature_abd_adj

# ------------------------------------------------------------
# Restore original orientation and save output
# ------------------------------------------------------------
adj_samp_by_feat <- t(adj_feat_by_samp)
stopifnot(identical(rownames(adj_samp_by_feat), rownames(abd)))

write.csv(
  adj_samp_by_feat,
  "nonchina_A50_MMUPHin.csv",
  quote = FALSE
)

message("MMUPHin batch correction completed. Output saved to nonchina_A50_MMUPHin.csv.")
