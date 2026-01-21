# Optimal Benchmark

This repository contains the codebase used to run large-scale benchmarking experiments for microbiome-based diagnostic modeling.

## Scope of the Benchmark

The benchmark systematically evaluates combinations of:

- Feature prevalence filtering strategies
- Batch-effect correction methods
- Machine learning algorithms
- Training strategies across heterogeneous cohorts

The focus is on understanding how methodological choices interact to influence model performance and stability.

## What This Repository Contains

This repository includes only the scripts required to train, evaluate, and benchmark models.

Raw sequencing data and processed abundance tables are not included. Input data are assumed to be generated from publicly available cohorts and provided in a standardized format.

## Usage

The benchmark is designed to be executed in a modular manner:

1. Prepare microbiome abundance tables
2. Apply feature filtering and batch-effect correction
3. Train models under different strategies
4. Evaluate and summarize performance metrics

Scripts are organized by functional modules rather than datasets.

## Model Outputs

For each model and training strategy, the benchmark generates two standardized output files based on repeated evaluations.

### Performance metrics

Model performance is evaluated on the validation dataset across multiple random seeds. The following file is generated:

- `validation_metrics_40times.csv`  
  This file records classification performance metrics for each repetition, including Accuracy, Precision, Recall, F1-score, and ROC-AUC. Each row corresponds to one random seed, enabling assessment of performance variability and stability.

### Feature importance

For models that provide intrinsic feature importance scores, the benchmark additionally records the most informative features:

- `top20_features_40times.csv`  
  This file contains the top 20 features ranked by model-specific importance scores for each repetition. For each random seed, feature rank, taxonomic name, and importance value are reported.

These outputs are designed to support both performance comparison and systematic analysis of feature-level stability across repeated runs.

## Graphical Abstract

![Graphical abstract summarizing the benchmarking framework](figures/graphical_abstract.jpg)

## Benchmark Workflow

![Overview of the benchmark workflow and training strategies](figures/workflow_overview.jpg)
