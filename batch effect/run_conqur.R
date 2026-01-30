#!/usr/bin/env Rscript
# ============================================================
# ConQuR batch-effect correction pipeline
# - Outputs corrected abundances for all batches
# - Reference batch retains original normalized abundances
# ============================================================

suppressPackageStartupMessages({
  library(ConQuR)
  library(data.table)
  library(dplyr)
  library(foreach)
})

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
abundance_file  <- "nonchina_A50_full.csv"    # abundance table (rows: samples, cols: taxa)
meta_file       <- "nonchina_A50_meta.csv"    # metadata with Sample, batch, and label
batch_ref_level <- "USA"                      # reference batch
batch_col       <- "group"                    # batch column in metadata
label_col       <- "label"                    # phenotype / outcome column
output_file     <- "nonchina_A50_conqur_fixed.csv"

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
abundance_table <- read.csv(
  abundance_file,
  row.names = 1,
  check.names = FALSE
)

meta_info <- read.csv(
  meta_file,
  stringsAsFactors = TRUE
)

# Align samples between abundance table and metadata
common_samples <- intersect(
  rownames(abundance_table),
  meta_info$Sample
)

abundance_table <- abundance_table[common_samples, , drop = FALSE]
meta_info <- meta_info[match(common_samples, meta_info$Sample), ]

# ------------------------------------------------------------
# Prepare variables for ConQuR
# ------------------------------------------------------------
batch <- as.factor(meta_info[[batch_col]])
covar <- data.frame(
  label = meta_info[[label_col]]
)

# Use sequential backend to avoid parallel instability
registerDoSEQ()

# ------------------------------------------------------------
# Run ConQuR batch correction
# ------------------------------------------------------------
corrected_abundance <- ConQuR(
  tax_tab   = as.matrix(abundance_table),
  batchid  = batch,
  covar    = covar,
  batch_ref = batch_ref_level
)

# ------------------------------------------------------------
# Post-processing
# ------------------------------------------------------------

# Set negative values to zero
corrected_abundance[corrected_abundance < 0] <- 0

# For reference batch: use original row-normalized abundances
ref_idx <- meta_info[[batch_col]] == batch_ref_level
corrected_abundance[ref_idx, ] <- sweep(
  as.matrix(abundance_table[ref_idx, ]),
  1,
  rowSums(as.matrix(abundance_table[ref_idx, ])),
  "/"
)

# Re-normalize each sample to sum to 1
corrected_abundance <- sweep(
  corrected_abundance,
  1,
  rowSums(corrected_abundance),
  "/"
)

# ------------------------------------------------------------
# Save output
# ------------------------------------------------------------
fwrite(
  as.data.table(corrected_abundance, keep.rownames = "Sample"),
  output_file,
  quote = FALSE,
  row.names = TRUE
)

message("ConQuR batch correction completed. Output file: ", output_file)
