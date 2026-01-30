#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_transfer_finetune_last.py

Description:
- Train an initial model on the training dataset
- Fine-tune the last layer using a transfer dataset
- Evaluate on an independent test dataset
- Automatically detect and transpose input tables if needed
- Align features across datasets
- Repeat experiments across multiple random seeds
"""

import os
import argparse
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ------------------------------------------------------------
# Utilities: reproducibility
# ------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ------------------------------------------------------------
# Utilities: robust CSV loader (auto-detect orientation)
# ------------------------------------------------------------
def try_load_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    cols_lower = [c.lower() for c in df.columns]

    # Case 1: explicit Sample / label columns
    if "sample" in cols_lower and "label" in cols_lower:
        return df

    # Case 2: features × samples, transpose required
    try:
        df2 = (
            df.set_index(df.columns[0])
              .T.reset_index()
              .rename(columns={"index": "Sample"})
        )
        df2.columns = [c.strip() for c in df2.columns]
        if any(c.lower() == "label" for c in df2.columns):
            return df2
    except Exception:
        pass

    # Case 3: label stored as a row
    first_col_vals = df.iloc[:, 0].astype(str).str.strip().str.lower()
    label_row_idx = next((i for i, v in enumerate(first_col_vals) if v == "label"), None)

    if label_row_idx is not None:
        df_indexed = df.set_index(df.columns[0])
        label_series = df_indexed.loc[
            df_indexed.index.str.strip().str.lower() == "label"
        ].iloc[0]

        df_features = df_indexed.loc[
            ~(df_indexed.index.str.strip().str.lower() == "label")
        ]

        df_t = df_features.T.reset_index().rename(columns={"index": "Sample"})
        df_t["label"] = df_t["Sample"].map(label_series)
        return df_t

    raise ValueError(
        f"Unable to infer table format for `{path}`. "
        "Please ensure Sample and label are provided."
    )

# ------------------------------------------------------------
# Feature alignment
# ------------------------------------------------------------
def align_test_to_train(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    def is_meta(col):
        return col.strip().lower() in ("sample", "label")

    train_feats = [c for c in train_df.columns if not is_meta(c)]
    test_feats = [c for c in test_df.columns if not is_meta(c)]

    train_norm = {c.strip().lower(): c for c in train_feats}
    test_norm = {c.strip().lower(): c for c in test_feats}

    test_df2 = test_df.copy()

    # Add missing features
    for feat_lower, feat_name in train_norm.items():
        if feat_lower in test_norm:
            if test_norm[feat_lower] != feat_name:
                test_df2.rename(columns={test_norm[feat_lower]: feat_name}, inplace=True)
        else:
            test_df2[feat_name] = 0.0

    # Remove extra features
    for feat_lower, feat_name in test_norm.items():
        if feat_lower not in train_norm:
            test_df2.drop(columns=[feat_name], inplace=True, errors="ignore")

    # Reorder columns
    cols_final = []
    if "sample" in [c.lower() for c in test_df2.columns]:
        cols_final.append(next(c for c in test_df2.columns if c.lower() == "sample"))
    if "label" in [c.lower() for c in test_df2.columns]:
        cols_final.append(next(c for c in test_df2.columns if c.lower() == "label"))
    cols_final += train_feats

    return test_df2[cols_final]

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ------------------------------------------------------------
# Model definition
# ------------------------------------------------------------
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims=(128, 64),
        dropout: float = 0.3,
        n_classes: int = 2
    ):
        super().__init__()

        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(last_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            last_dim = h

        layers.append(nn.Linear(last_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_prob=None):
    results = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "auc": None
    }

    try:
        if y_prob is not None and y_prob.shape[1] == 2:
            results["auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
    except Exception:
        pass

    return results

# ------------------------------------------------------------
# Training + fine-tuning (last layer only)
# ------------------------------------------------------------
def run_seed(
    seed,
    X_train, y_train,
    X_transfer, y_transfer,
    X_val, y_val,
    input_dim, n_classes,
    hidden_dims, device,
    batch_size, epochs, ft_epochs
):
    set_seed(seed)

    model = MLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        n_classes=n_classes
    ).to(device)

    # Freeze all layers except the final classifier
    for param in model.net[:-1].parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TabularDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )
    transfer_loader = DataLoader(
        TabularDataset(X_transfer, y_transfer),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        TabularDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False
    )

    # Initial training
    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    # Fine-tuning on transfer set
    model.train()
    for _ in range(ft_epochs):
        for xb, yb in transfer_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device)
            out = model(xb)
            p = torch.softmax(out, dim=1).cpu().numpy()
            probs.append(p)
            preds.append(np.argmax(p, axis=1))

    probs = np.vstack(probs)
    preds = np.concatenate(preds)

    metrics = compute_metrics(y_val, preds, probs)
    return metrics, preds, probs

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(args):
    device = torch.device("cpu")
    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(",") if x.strip())

    print("Loading training set:", args.train)
    train_df = try_load_table(args.train)

    print("Loading transfer set:", args.transfer)
    transfer_df = try_load_table(args.transfer)

    print("Loading validation set:", args.test)
    val_df = try_load_table(args.test)

    transfer_df = align_test_to_train(train_df, transfer_df)
    val_df = align_test_to_train(train_df, val_df)

    feat_cols = [c for c in train_df.columns if c.lower() not in ("sample", "label")]

    X_train = train_df[feat_cols].astype(float).values
    X_transfer = transfer_df[feat_cols].astype(float).values
    X_val = val_df[feat_cols].astype(float).values

    y_train_raw = train_df["label"].astype(str).values
    y_transfer_raw = transfer_df["label"].astype(str).values
    y_val_raw = val_df["label"].astype(str).values

    le = LabelEncoder().fit(y_train_raw)
    y_train = le.transform(y_train_raw)
    y_transfer = le.transform(y_transfer_raw)
    y_val = le.transform(y_val_raw)

    input_dim = X_train.shape[1]
    n_classes = len(le.classes_)

    train_name = os.path.splitext(os.path.basename(args.train))[0]
    outdir = os.path.join(args.outdir, train_name)
    os.makedirs(outdir, exist_ok=True)

    all_metrics = []

    for seed in range(args.repeat):
        print(f"Running seed {seed}")
        metrics, preds, _ = run_seed(
            seed,
            X_train, y_train,
            X_transfer, y_transfer,
            X_val, y_val,
            input_dim, n_classes,
            hidden_dims, device,
            args.batch_size,
            args.epochs,
            args.ft_epochs
        )
        metrics["seed"] = seed
        all_metrics.append(metrics)

        np.savetxt(
            os.path.join(outdir, f"preds_seed{seed}.csv"),
            preds,
            delimiter=",",
            fmt="%d"
        )

    pd.DataFrame(all_metrics).to_csv(
        os.path.join(outdir, "transfer_metrics_40seeds.csv"),
        index=False
    )

    print("Metrics saved to:", os.path.join(outdir, "transfer_metrics_40seeds.csv"))

# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--transfer", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--outdir", default="transfer_results")
    parser.add_argument("--hidden_dims", default="128,64")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--ft_epochs", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=40)

    main(parser.parse_args())
