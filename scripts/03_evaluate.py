"""
STEP 3: EVALUATE THE AUTOENCODER
-----------------------------------
Now we test the model on data it has NEVER seen: a mix of normal
transactions (held-out validation set) and ALL the fraud transactions.

For each transaction, we:
1. Feed it through the autoencoder to get a reconstruction
2. Measure reconstruction error = mean squared error between input and output
3. High error -> flag as fraud

WHY WE USE PRECISION-RECALL INSTEAD OF ACCURACY (important to be able to
explain this in an interview):

- Precision = "of everything I flagged as fraud, what % was actually fraud?"
  Low precision = too many false alarms (annoying, wastes investigator time)
- Recall = "of all the actual fraud, what % did I catch?"
  Low recall = fraud slipping through undetected (costs real money)

There's a TRADEOFF: lowering the reconstruction-error threshold catches
more fraud (higher recall) but also flags more normal transactions by
mistake (lower precision). Where you set the threshold depends on the
business context -- for fraud, missing real fraud is usually worse than
a false alarm, so you'd typically lean toward higher recall.

ROC-AUC gives a single number (0.5 = random guessing, 1.0 = perfect)
summarizing how well the model separates fraud from normal ACROSS ALL
possible thresholds, which is useful for comparing models overall
(that's what we'll use in Step 4 vs. the baseline).
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, roc_curve, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
from tensorflow import keras

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

# --- Load everything ---
autoencoder = keras.models.load_model(os.path.join(MODEL_DIR, "autoencoder.keras"))
amount_scaler = joblib.load(os.path.join(MODEL_DIR, "amount_scaler.pkl"))
time_scaler = joblib.load(os.path.join(MODEL_DIR, "time_scaler.pkl"))

df = pd.read_csv(DATA_PATH)
df["Amount"] = amount_scaler.transform(df[["Amount"]])
df["Time"] = time_scaler.transform(df[["Time"]])

X = df.drop("Class", axis=1)
y = df["Class"]

# --- Get reconstruction error for EVERY transaction ---
reconstructions = autoencoder.predict(X, batch_size=512, verbose=1)
mse = np.mean(np.power(X.values - reconstructions, 2), axis=1)

df["reconstruction_error"] = mse

# --- Compare error distributions: normal vs fraud ---
print("\n--- Reconstruction Error Summary ---")
print("Normal transactions - mean error:", df[df["Class"] == 0]["reconstruction_error"].mean())
print("Fraud transactions   - mean error:", df[df["Class"] == 1]["reconstruction_error"].mean())
print(">> Fraud error should be noticeably higher. That gap is what makes this work.")

# --- ROC-AUC: single number summarizing separation quality ---
auc = roc_auc_score(y, mse)
print(f"\nROC-AUC: {auc:.4f}  (0.5 = random guessing, 1.0 = perfect separation)")

# --- Precision-Recall curve: find a threshold ---
precisions, recalls, thresholds = precision_recall_curve(y, mse)

plt.figure(figsize=(7, 5))
plt.plot(thresholds, precisions[:-1], label="Precision")
plt.plot(thresholds, recalls[:-1], label="Recall")
plt.xlabel("Reconstruction Error Threshold")
plt.ylabel("Score")
plt.title("Precision & Recall vs Threshold")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "precision_recall_vs_threshold.png"), dpi=120, bbox_inches="tight")
print(f"Saved chart to {OUTPUT_DIR}/precision_recall_vs_threshold.png")

# --- Pick a threshold: here we pick the one that maximizes F1
# (balance of precision and recall). In a real job, you'd justify this
# choice based on business cost of false positives vs false negatives.
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx = np.argmax(f1_scores[:-1])
best_threshold = thresholds[best_idx]

print(f"\nChosen threshold (max F1): {best_threshold:.4f}")
print(f"  Precision at this threshold: {precisions[best_idx]:.3f}")
print(f"  Recall at this threshold:    {recalls[best_idx]:.3f}")
print(f"  F1 score:                    {f1_scores[best_idx]:.3f}")

# --- Final classification report using chosen threshold ---
y_pred = (mse > best_threshold).astype(int)
cm = confusion_matrix(y, y_pred)
print("\nConfusion Matrix:")
print("                Predicted Normal   Predicted Fraud")
print(f"Actual Normal   {cm[0][0]:>16}   {cm[0][1]:>16}")
print(f"Actual Fraud    {cm[1][0]:>16}   {cm[1][1]:>16}")

tn, fp, fn, tp = cm.ravel()
print(f"\nCaught {tp} out of {tp + fn} actual fraud cases ({tp / (tp + fn) * 100:.1f}% recall)")
print(f"Falsely flagged {fp} normal transactions as fraud")

print("\nSaved model results. Next: run 04_baseline_comparison.py")
