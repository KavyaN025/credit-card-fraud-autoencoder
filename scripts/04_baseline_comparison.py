"""
STEP 4: BASELINE COMPARISON
------------------------------
This is the part most tutorial projects SKIP -- and it's exactly what
makes this project defensible instead of "I copied a Kaggle notebook."

We compare the autoencoder against Isolation Forest, a simpler,
well-established anomaly detection algorithm that works completely
differently: instead of reconstruction error, it isolates anomalies by
randomly partitioning the data -- anomalies get isolated in fewer
partitions (they're "easier to separate" from the rest of the data).

WHY THIS MATTERS: if the autoencoder doesn't clearly beat a simpler
method, that's worth knowing (and saying honestly) rather than hiding it.
If it DOES win, you now have a real, defensible reason to justify the
extra complexity of a neural network over a simpler classical method --
which is exactly the kind of judgment call interviewers want to see you
reason through.
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    precision_recall_curve
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

amount_scaler = joblib.load(os.path.join(MODEL_DIR, "amount_scaler.pkl"))
time_scaler = joblib.load(os.path.join(MODEL_DIR, "time_scaler.pkl"))
df = pd.read_csv(DATA_PATH)
df["Amount"] = amount_scaler.transform(df[["Amount"]])
df["Time"] = time_scaler.transform(df[["Time"]])

X = df.drop("Class", axis=1)
y = df["Class"]

fraud_rate = y.mean()
iso_forest = IsolationForest(contamination=fraud_rate, random_state=42, n_jobs=-1)
iso_forest.fit(X)

iso_scores = -iso_forest.score_samples(X)

iso_auc = roc_auc_score(y, iso_scores)

precisions, recalls, thresholds = precision_recall_curve(y, iso_scores)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx = np.argmax(f1_scores[:-1])
best_threshold = thresholds[best_idx]

iso_pred_binary = (iso_scores > best_threshold).astype(int)
iso_precision = precision_score(y, iso_pred_binary)
iso_recall = recall_score(y, iso_pred_binary)
iso_f1 = f1_score(y, iso_pred_binary)

print("=== Isolation Forest (baseline, threshold tuned for fair comparison) ===")
print(f"Chosen threshold (max F1): {best_threshold:.4f}")
print(f"ROC-AUC:   {iso_auc:.4f}")
print(f"Precision: {iso_precision:.3f}")
print(f"Recall:    {iso_recall:.3f}")
print(f"F1 Score:  {iso_f1:.3f}")
print("\n>> Compare these numbers against the autoencoder's results from")
print(">> Step 3 (03_evaluate.py output). Whichever wins on ROC-AUC and F1")
print(">> is the model you'd recommend -- and you should be able to explain")
print(">> WHY, not just which number is bigger. E.g.: does the autoencoder")
print(">> capture non-linear relationships in the PCA'd features that")
print(">> Isolation Forest's tree-based splits miss? Write your reasoning")
print(">> down in the README once you see the actual numbers.")
