# XGBoost.py
import pandas as pd
import numpy as np
import os
import argparse
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ======== 命令行参数 ========
parser = argparse.ArgumentParser(description="XGBoost微生物属分类建模")
parser.add_argument("--train", required=True, help="训练集 CSV 文件路径")
parser.add_argument("--test", required=True, help="测试集 CSV 文件路径")
parser.add_argument("--val", required=True, help="验证集 CSV 文件路径")
parser.add_argument("--output", default="XGBoost_results", help="输出文件夹")
parser.add_argument("--repeat", type=int, default=40, help="重复次数")
args = parser.parse_args()

OUTPUT_DIR = args.output
REPEAT = args.repeat
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======== 读取数据 ========
train_df = pd.read_csv(args.train, index_col=0)
test_df  = pd.read_csv(args.test, index_col=0)
val_df   = pd.read_csv(args.val, index_col=0)

# ======== 分离特征与标签 ========
X_train = train_df.drop(columns=["label"]).apply(pd.to_numeric, errors='coerce').fillna(0)
y_train = train_df["label"]
X_test  = test_df.drop(columns=["label"]).apply(pd.to_numeric, errors='coerce').fillna(0)
y_test  = test_df["label"]
X_val   = val_df.drop(columns=["label"]).apply(pd.to_numeric, errors='coerce').fillna(0)
y_val   = val_df["label"]

# ======== 清洗列名，替换特殊字符 ========
special_chars = r"[\[\]<>(){}]"
X_train.columns = X_train.columns.astype(str).str.replace(special_chars, "_", regex=True)
X_test.columns  = X_test.columns.astype(str).str.replace(special_chars, "_", regex=True)
X_val.columns   = X_val.columns.astype(str).str.replace(special_chars, "_", regex=True)

# ======== 列对齐 ========
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
X_val  = X_val.reindex(columns=X_train.columns, fill_value=0)

# ======== 循环训练 ========
metrics_all = []
features_all = []

for i in range(REPEAT):
    seed = i
    clf = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=seed
    )
    clf.fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))
    
    y_pred = clf.predict(X_val)
    y_prob = clf.predict_proba(X_val)[:,1]
    
    # 保存指标
    metrics_all.append({
        "Seed": seed,
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred, zero_division=0),
        "Recall": recall_score(y_val, y_pred, zero_division=0),
        "F1": f1_score(y_val, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_val, y_prob)
    })
    
    # 前20特征
    importance = clf.feature_importances_
    feature_names = X_train.columns.tolist()
    top_idx = np.argsort(importance)[::-1][:20]
    features_all.append(pd.DataFrame({
        "Seed": seed,
        "Rank": np.arange(1,21),
        "Genus": np.array(feature_names)[top_idx],
        "Importance": importance[top_idx]
    }))

# ======== 保存结果 ========
pd.DataFrame(metrics_all).to_csv(os.path.join(OUTPUT_DIR, "validation_metrics_40times.csv"), index=False)
pd.concat(features_all, ignore_index=True).to_csv(os.path.join(OUTPUT_DIR, "top20_features_40times.csv"), index=False)

print(f"✅ XGBoost 完成，结果已保存到 {OUTPUT_DIR}")
