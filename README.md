# Optimal Benchmark

This repository contains scripts for benchmarking gut microbiome diagnostic models using genus (or taxon) abundance features. The focus is on methodological choices and their impact on classification performance and stability. The text below is conservative and limited to what is present in the repository.

## Project Purpose

- Compare how filtering strategies, batch-effect correction, model choice, and training strategy influence classification performance.
- Produce reproducible model evaluation results with multiple random seeds, with emphasis on ROC_AUC.

## Scope of the Benchmark

The benchmark systematically evaluates combinations of:

- Feature prevalence filtering strategies
- Batch-effect correction methods
- Machine learning algorithms
- Training strategies across heterogeneous cohorts

## Data and Results

- This repository includes scripts and a sample input file only.
- Raw sequencing data and full processing pipelines are not included.
- Example input data: `filter/15_genus_abundance_with_group.csv` (original data file).

## Reproducibility and Parameter Traceability

- Most scripts accept command-line arguments for input, output, and the number of repeats (default 40).
- Random seeds are fixed as `0..repeat-1`, and each repeat is recorded independently.
- Test/validation features are aligned to training features, with missing columns filled as 0.

## Directory Structure

```
D:\Optimal-Benchmark
├─ batch effect\               # Batch-effect correction (R)
├─ data fusion\                # Data-fusion training strategy (Python)
├─ figures\                    # Workflow figures
├─ filter\                     # Feature filtering and example data
├─ traditional method\         # Traditional train-test strategy (Python)
├─ transfer learning\          # Transfer learning (Python / PyTorch)
└─ README.md
```

## Main Scripts

### filter/2.py - Group split and feature filtering

- Input: CSV with `Sample`, `label`, `group` columns; all other columns are features.
- `group` is geographic/batch information (used for batch correction).
- `label` is biological status (0 = healthy, 1 = disease) for binary classification.
- Splits samples into China and non-China based on `group`:
  - China: no feature removal, only row-wise normalization (TSS).
  - non-China:
    - Global prevalence filtering (A50 / A90)
    - Project-wise prevalence filtering (B50 / B90; project ID is the prefix before `_` in `Sample`)
- Outputs:
  - `china_full.csv / china_label.csv / china_label_group.csv / china_meta.csv`
  - `nonchina_A50/...`, `nonchina_A90/...`, `nonchina_B50/...`, `nonchina_B90/...`
  - Each non-China directory contains `*_removed_full.csv`, `*_removed_label.csv`, `*_removed_label_group.csv`, `*_removed_meta.csv`
  - `summary_removed.csv` summarizes removed features and sample counts
- Note: `removed_*` files represent the removed feature set. Both the kept set and removed set are modeled through the same pipeline for comparison.

### batch effect/*.R - Batch-effect correction (fixed filenames)

These scripts use fixed input/output filenames and expect to run in a working directory that contains those files (adjust paths if needed):

- `run_combat.R` (sva::ComBat)
  - Input: `nonchina_A50_full.csv` + `nonchina_A50_meta.csv`
  - Output: `nonchina_A50_combat.csv`
- `run_conqur.R` (ConQuR)
  - Input: `nonchina_A50_full.csv` + `nonchina_A50_meta.csv`
  - Output: `nonchina_A50_conqur_fixed.csv`
- `run_mmuphin.R` (MMUPHin::adjust_batch)
  - Input: `nonchina_A50_full.csv` + `nonchina_A50_meta.csv`
  - Output: `nonchina_A50_MMUPHin.csv` (and `nonchina_A50_MMUPHin_diagnostic.pdf`)

Common behavior:
- `group` is treated as the batch variable; `label` is preserved as biological signal.
- If inputs are not row-normalized, total-sum scaling is applied.

### traditional method/*.py - Traditional train-test strategy

Models included:
`AdaBoost, CatBoost, ExtraTrees, GaussianNB, GradientBoosting, KNN, LightGBM, LogisticRegression, MLP, QuadraticDiscriminantAnalysis, RF, SVC, XGBoost`

- Arguments: `--train`, `--test`, `--output`, `--repeat`
- CSVs are read with `index_col=0`; `label` is required, all other columns are features.
- Test features are aligned to training features (missing columns filled as 0).
- Output per model (in `--output` directory):
  - `<MODEL>_metrics_40times.csv` (includes ROC_AUC; some scripts also include PR_AUC)
  - `<MODEL>_top20_features_40times.csv`
- Emphasis is on ROC_AUC; model performance is evaluated across 40 random seeds.

### data fusion/*.py - Data-fusion training strategy

Models included:
`AdaBoost, CatBoost, ExtraTrees, GaussianNB, GradientBoosting, KNN, LightGBM, LogisticRegression, MLP, QDA, RF, SVC, XGBoost`

- Arguments: `--train`, `--test`, `--val`, `--output`, `--repeat`
- Training uses `train + test` combined; evaluation is on `val`.
- Output per model (in `--output` directory):
  - `validation_metrics_40times.csv`
  - `top20_features_40times.csv`
- Emphasis is on ROC_AUC; model performance is evaluated across 40 random seeds.

### transfer learning/train_transfer_finetune_last.py - Transfer learning

- Arguments: `--train`, `--transfer`, `--test`, `--outdir`, `--hidden_dims`, `--batch_size`, `--epochs`, `--ft_epochs`, `--repeat`
- Input format auto-detection:
  - Supports sample-by-feature tables with `Sample` and `label` columns
  - Supports transposed tables or label-as-row formats
- Features are aligned to the training set.
- Model: MLP; only the last layer is fine-tuned; default CPU execution.
- Output (`--outdir/<train_basename>/`):
  - `transfer_metrics_40seeds.csv`
  - `preds_seed{seed}.csv`

## Input Data Format

### Example 1: filter/2.py input (with group)

```
Sample,label,group,GenusA,GenusB,GenusC
P1_USA_001,0,USA,12,0,5
P2_CHN_003,1,China,0,7,3
```

- `group` is batch/region.
- `label` is binary biological status (0 = healthy, 1 = disease).
- Feature columns are taxa abundances (counts or relative abundances).

### Example 2: traditional/data-fusion scripts input (index_col=0)

```
Sample,label,GenusA,GenusB,GenusC
S1,0,0.12,0.00,0.05
S2,1,0.00,0.07,0.03
```

- First column is read as the index.
- `label` is required; all other columns are numeric features.

## Outputs

### Filtering outputs

- `china_*.csv`: China subset with row normalization.
- `nonchina_*` directories: removed feature sets and metadata for each threshold strategy.
- `summary_removed.csv`: removed features and sample count changes.

### Batch-effect correction outputs

- `nonchina_A50_combat.csv`
- `nonchina_A50_conqur_fixed.csv`
- `nonchina_A50_MMUPHin.csv` (plus diagnostic PDF)

### Traditional method outputs

- `<MODEL>_metrics_40times.csv` (ROC_AUC per seed)
- `<MODEL>_top20_features_40times.csv`

### Data-fusion outputs

- `validation_metrics_40times.csv`
- `top20_features_40times.csv`

### Transfer learning outputs

- `transfer_metrics_40seeds.csv`
- `preds_seed{seed}.csv`

## Dependencies (from imports)

- Python: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `torch`
- R: `sva`, `ConQuR`, `MMUPHin`, `data.table`, `dplyr`, `foreach`

## Graphical Abstract

![Graphical abstract summarizing the benchmarking framework](figures/graphical_abstract.jpg)

## Benchmark Workflow

![Overview of the benchmark workflow and training strategies](figures/workflow_overview.jpg)
