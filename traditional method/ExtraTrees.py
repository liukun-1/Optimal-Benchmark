import pandas as pd
import numpy as np
import os
import argparse
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc
)

parser = argparse.ArgumentParser(description="ExtraTrees微生物属分类建模")
parser.add_argument("--train", required=True)
parser.add_argument("--test", required=True)
parser.add_argument("--output", default="con_results_ExtraTrees")
parser.add_argument("--repeat", type=int, default=40)
args = parser.parse_args()

train_file = args.train
test_file = args.test
output_dir = args.output
repeat = args.repeat
os.makedirs(output_dir, exist_ok=True)
model_name = "ExtraTrees"

train_df = pd.read_csv(train_file, index_col=0)
test_df = pd.read_csv(test_file, index_col=0)
y_train = train_df["label"]
y_test = test_df["label"]
X_train = train_df.drop(columns=["label"])
X_test = test_df.drop(columns=["label"])
X_train.columns = X_train.columns.astype(str).str.replace(r"[\[\]<>(){}]", "_", regex=True)
X_test.columns = X_test.columns.astype(str).str.replace(r"[\[\]<>(){}]", "_", regex=True)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

results = []
top_features_all = []

for i in range(repeat):
    seed = i
    clf = ExtraTreesClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    precisions, recalls, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recalls, precisions)

    results.append({
        "Iteration": i+1, "Seed": seed,
        "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1": f1,
        "ROC_AUC": roc_auc, "PR_AUC": pr_auc
    })

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    top_features = pd.DataFrame({
        "Iteration": i+1, "Seed": seed,
        "Rank": np.arange(1,21),
        "Genus": X_train.columns[indices],
        "Importance": importances[indices]
    })
    top_features_all.append(top_features)

pd.DataFrame(results).to_csv(os.path.join(output_dir,f"{model_name}_metrics_40times.csv"), index=False)
pd.concat(top_features_all, ignore_index=True).to_csv(os.path.join(output_dir,f"{model_name}_top20_features_40times.csv"), index=False)
print(f"✅ {model_name} 结果已保存到 {output_dir}")
