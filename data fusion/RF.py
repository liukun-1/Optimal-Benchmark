# RandomForest.py
import pandas as pd
import numpy as np
import os
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument("--train", required=True)
parser.add_argument("--test", required=True)
parser.add_argument("--val", required=True)
parser.add_argument("--output", default="RandomForest")
parser.add_argument("--repeat", type=int, default=40)
args = parser.parse_args()

OUTPUT_DIR = args.output
REPEAT = args.repeat
os.makedirs(OUTPUT_DIR, exist_ok=True)

train_df = pd.read_csv(args.train, index_col=0)
test_df  = pd.read_csv(args.test, index_col=0)
val_df   = pd.read_csv(args.val, index_col=0)

X_train = train_df.drop(columns=["label"]).apply(pd.to_numeric, errors='coerce').fillna(0)
y_train = train_df["label"]
X_test  = test_df.drop(columns=["label"]).apply(pd.to_numeric, errors='coerce').fillna(0)
y_test  = test_df["label"]
X_val   = val_df.drop(columns=["label"]).apply(pd.to_numeric, errors='coerce').fillna(0)
y_val   = val_df["label"]

train_cols = X_train.columns.tolist()
X_test = X_test.reindex(columns=train_cols, fill_value=0)
X_val  = X_val.reindex(columns=train_cols, fill_value=0)

metrics_all = []
features_all = []

for i in range(REPEAT):
    seed = i
    clf = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
    clf.fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))
    
    y_pred = clf.predict(X_val)
    y_prob = clf.predict_proba(X_val)[:,1]
    
    metrics_all.append({
        "Seed": seed,
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred, zero_division=0),
        "Recall": recall_score(y_val, y_pred, zero_division=0),
        "F1": f1_score(y_val, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_val, y_prob)
    })
    
    importance = clf.feature_importances_
    feature_names = X_train.columns.tolist()
    top_idx = np.argsort(importance)[::-1][:20]
    features_all.append(pd.DataFrame({
        "Seed": seed,
        "Rank": np.arange(1,21),
        "Genus": np.array(feature_names)[top_idx],
        "Importance": importance[top_idx]
    }))

pd.DataFrame(metrics_all).to_csv(f"{OUTPUT_DIR}/validation_metrics_40times.csv", index=False)
pd.concat(features_all, ignore_index=True).to_csv(f"{OUTPUT_DIR}/top20_features_40times.csv", index=False)
print(f"✅ RandomForest 完成")
