#!/usr/bin/env Rscript
# ============================================================
# ComBat batch-effect correction pipeline (sva)
# - Automatically detects whether input is already normalized
# - Preserves biological signal via design matrix
# ============================================================

suppressPackageStartupMessages({
  library(sva)
})

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
abd <- read.csv(
  "nonchina_A50_full.csv",
  row.names = 1,
  check.names = FALSE
)

meta <- read.csv(
  "nonchina_A50_meta.csv",
  row.names = 1,
  check.names = FALSE
)

# ------------------------------------------------------------
# Basic validation
# ------------------------------------------------------------
if (!all(sapply(abd, is.numeric))) {
  stop("Abundance table contains non-numeric columns. Please retain taxa abundances only.")
}

if (any(is.na(as.matrix(abd)))) {
  warning("NA values detected and will be set to 0.")
  abd[is.na(abd)] <- 0
}

if (any(abd < 0)) {
  stop("Negative values detected. Input should be non-negative counts or relative abundances.")
}

# ------------------------------------------------------------
# Check whether data are already row-normalized (row sum ≈ 1)
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
# Prepare input for ComBat (features × samples)
# ------------------------------------------------------------
feat_by_samp <- t(as.matrix(abd))

meta$group <- as.factor(meta$group)
meta$label <- as.factor(meta$label)

# Align sample order
common_samples <- intersect(colnames(feat_by_samp), rownames(meta))
feat_by_samp <- feat_by_samp[, common_samples, drop = FALSE]
meta <- meta[common_samples, , drop = FALSE]

# ------------------------------------------------------------
# Define ComBat parameters
# ------------------------------------------------------------
batch <- meta$group                # batch effect to remove
biological_group <- meta$label     # biological signal to preserve

# Design matrix
mod <- model.matrix(~ biological_group)

# ------------------------------------------------------------
# Run ComBat
# ------------------------------------------------------------
combat_corrected <- ComBat(
  dat = feat_by_samp,
  batch = batch,
  mod = mod,
  par.prior = TRUE,
  prior.plots = FALSE
)

# ------------------------------------------------------------
# Restore original orientation (samples × features)
# ------------------------------------------------------------
corrected_abd <- t(combat_corrected)

# ------------------------------------------------------------
# Save output
# ------------------------------------------------------------
write.csv(
  corrected_abd,
  "nonchina_A50_combat.csv",
  quote = FALSE,
  row.names = TRUE
)

message("ComBat batch correction completed. Output saved to nonchina_A50_combat.csv.")
