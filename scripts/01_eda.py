"""
STEP 1: EXPLORATORY DATA ANALYSIS
-----------------------------------
Goal: understand the data BEFORE building anything. Specifically, see
just how imbalanced the classes are -- this single fact is why the whole
project uses autoencoders instead of a normal classifier.

Dataset: creditcard.csv from Kaggle
(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Contains real European card transactions from Sept 2013.
- Features V1-V28 are already PCA-transformed (anonymized) for privacy --
  you don't need to know what they "mean", just treat them as numeric
  features.
- 'Time' = seconds since the first transaction in the dataset.
- 'Amount' = transaction amount.
- 'Class' = 1 for fraud, 0 for normal. THIS is what we're trying to catch.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

df = pd.read_csv(DATA_PATH)

print("Shape of dataset:", df.shape)
print("\nColumn names:", list(df.columns))

# THE KEY NUMBER FOR THIS ENTIRE PROJECT
class_counts = df["Class"].value_counts()
fraud_pct = (class_counts[1] / len(df)) * 100

print(f"\nNormal transactions: {class_counts[0]}")
print(f"Fraudulent transactions: {class_counts[1]}")
print(f"Fraud percentage: {fraud_pct:.3f}%")
print(
    "\n>> THIS is why 'accuracy' is a useless metric here. "
    "A model that predicts 'not fraud' for EVERY transaction would "
    f"still be {100 - fraud_pct:.2f}% accurate, while catching zero fraud. "
    "This is the whole reason we need precision/recall instead of accuracy."
)

# Visualize the imbalance
plt.figure(figsize=(6, 4))
sns.countplot(x="Class", data=df)
plt.title(f"Class Distribution (Fraud = {fraud_pct:.3f}% of all transactions)")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.savefig(os.path.join(OUTPUT_DIR, "class_imbalance.png"), dpi=120, bbox_inches="tight")
print(f"\nSaved chart to {OUTPUT_DIR}/class_imbalance.png")

# Look at transaction amount by class -- do fraud amounts look different?
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.histplot(df[df["Class"] == 0]["Amount"], bins=50, color="steelblue")
plt.title("Normal Transaction Amounts")
plt.xlim(0, 500)

plt.subplot(1, 2, 2)
sns.histplot(df[df["Class"] == 1]["Amount"], bins=50, color="crimson")
plt.title("Fraudulent Transaction Amounts")
plt.xlim(0, 500)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "amount_distribution.png"), dpi=120, bbox_inches="tight")
print(f"Saved chart to {OUTPUT_DIR}/amount_distribution.png")

print("\nDone. Open the two PNGs in outputs/ and look at them before moving to Step 2.")
