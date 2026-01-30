#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split Chinese and non-Chinese samples and perform feature filtering.

Workflow:
- Split samples into China and Non-China groups based on the `group` column
- China samples: no feature removal, only row-wise normalization
- Non-China samples: feature filtering under multiple strategies
    * Global prevalence filtering (A50 / A90)
    * Cohort-wise prevalence filtering (B50 / B90)
- Export multiple data versions and a summary of removed features
"""

import os
import argparse
import pandas as pd


# -----------------------------------------------------------
# I/O utilities
# ------------------------------------------------------------
def load_input(path):
    df = pd.read_csv(path)
    for c in ["Sample", "label", "group"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    return df


def detect_abund_cols(df):
    return [c for c in df.columns if c not in ["Sample", "label", "group"]]


# ------------------------------------------------------------
# Feature filtering strategies
# ------------------------------------------------------------
def filter_global(df, thr):
    """Global prevalence filtering across all samples."""
    abund = detect_abund_cols(df)
    ratio = (df[abund] > 0).sum(axis=0) / df.shape[0]
    kept = ratio[ratio > thr].index.tolist()
    return [c for c in abund if c not in kept]


def filter_project(df, thr):
    """Project-wise prevalence filtering (union across projects)."""
    abund = detect_abund_cols(df)
    tmp = df.copy()
    tmp["Project"] = tmp["Sample"].str.split("_", n=1).str[0]

    kept = set()
    for _, sub in tmp.groupby("Project"):
        ratio = (sub[abund] > 0).sum(axis=0) / sub.shape[0]
        kept.update(ratio[ratio > thr].index.tolist())

    return [c for c in abund if c not in kept]


# ------------------------------------------------------------
# Sample filtering and normalization
# ------------------------------------------------------------
def drop_zero_samples(df, cols):
    before = df.shape[0]
    after = df.loc[df[cols].sum(axis=1) > 0].copy()
    return after, before, after.shape[0]


def row_normalize(df, meta_cols):
    abund = df.drop(columns=meta_cols, errors="ignore")
    sums = abund.sum(axis=1)
    abund_norm = abund.div(sums, axis=0).fillna(0)
    return pd.concat([df[meta_cols], abund_norm], axis=1)


# ------------------------------------------------------------
# Export utilities
# ------------------------------------------------------------
def export_versions(df, abund_cols, outdir, prefix, normalize=True):
    os.makedirs(outdir, exist_ok=True)

    def save(subdf, meta_cols, name):
        d = subdf.copy()
        if normalize:
            d = row_normalize(d, meta_cols)
        d.to_csv(os.path.join(outdir, f"{prefix}_{name}.csv"), index=False)

    save(df[["Sample"] + abund_cols], ["Sample"], "removed_full")
    save(df[["Sample", "label"] + abund_cols], ["Sample", "label"], "removed_label")
    save(
        df[["Sample", "label", "group"] + abund_cols],
        ["Sample", "label", "group"],
        "removed_label_group"
    )

    df[["Sample", "label", "group"]].to_csv(
        os.path.join(outdir, f"{prefix}_removed_meta.csv"),
        index=False
    )


def export_china(df_china, outdir):
    cols = detect_abund_cols(df_china)
    os.makedirs(outdir, exist_ok=True)

    def save(subdf, meta_cols, name):
        d = row_normalize(subdf, meta_cols)
        d.to_csv(os.path.join(outdir, f"china_{name}.csv"), index=False)

    save(df_china[["Sample"] + cols], ["Sample"], "full")
    save(df_china[["Sample", "label"] + cols], ["Sample", "label"], "label")
    save(
        df_china[["Sample", "label", "group"] + cols],
        ["Sample", "label", "group"],
        "label_group"
    )

    df_china[["Sample", "label", "group"]].to_csv(
        os.path.join(outdir, "china_meta.csv"),
        index=False
    )


# ------------------------------------------------------------
# Non-China processing
# ------------------------------------------------------------
def process_nonchina(df, outdir, summary):
    for thr, thr_label in [(0.5, "50"), (0.9, "90")]:
        for mode, prefix in [("global", "A"), ("project", "B")]:
            removed = (
                filter_global(df, thr)
                if mode == "global"
                else filter_project(df, thr)
            )

            if not removed:
                print(f"[Non-China] {prefix}{thr_label}: no features removed")
                continue

            sub = df[["Sample", "label", "group"] + removed].copy()
            sub, n_before, n_after = drop_zero_samples(sub, removed)

            pref = f"nonchina_{prefix}{thr_label}"
            export_versions(sub, removed, os.path.join(outdir, pref), pref)

            summary.append(dict(
                dataset=pref,
                mode=mode,
                threshold=thr,
                n_genera=len(removed),
                n_samples_before=n_before,
                n_samples_after=n_after,
                genera=";".join(removed)
            ))


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Split China / Non-China samples and perform feature filtering"
    )
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    df = load_input(args.input)

    china_df = df[df["group"].str.lower() == "china"].copy()
    non_df = df[df["group"].str.lower() != "china"].copy()

    os.makedirs(args.outdir, exist_ok=True)
    summary = []

    if not china_df.empty:
        export_china(china_df, args.outdir)

    if not non_df.empty:
        process_nonchina(non_df, args.outdir, summary)

    if summary:
        pd.DataFrame(summary).to_csv(
            os.path.join(args.outdir, "summary_removed.csv"),
            index=False
        )

    print("Processing completed:", args.outdir)


if __name__ == "__main__":
    main()
