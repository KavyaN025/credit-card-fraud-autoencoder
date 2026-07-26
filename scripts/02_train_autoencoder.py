"""
STEP 2: TRAIN THE AUTOENCODER
--------------------------------
THE CORE IDEA (understand this before running anything):

An autoencoder is a neural network trained to do something that sounds
pointless: reconstruct its own input. It has two halves:

  Input -> [ENCODER] -> small bottleneck -> [DECODER] -> Output
  (30 features)         (e.g. 8 numbers)                (30 features)

Because the bottleneck is much smaller than the input, the network CANNOT
just copy the input through. It's forced to compress the data down to
only the most essential patterns, then reconstruct from that compressed
version. This only works well if the network has actually learned the
*typical structure* of the data.

THE TRICK FOR FRAUD DETECTION:
We train the autoencoder ONLY on normal (non-fraud) transactions. It
becomes very good at reconstructing normal transactions with low error.

When we later feed it a FRAUDULENT transaction (which it has never seen),
the patterns are different from what it learned -> it does a BAD job of
reconstructing it -> the reconstruction error (difference between input
and output) is HIGH.

So: high reconstruction error = "this looks unusual" = likely fraud.
We never explicitly tell the model what fraud looks like. It flags
ANYTHING that deviates from "normal" -- that's why this is called
unsupervised anomaly detection.
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

df = pd.read_csv(DATA_PATH)

# --- Preprocessing ---
# V1-V28 are already PCA-transformed (roughly standardized already).
# Time and Amount are NOT scaled -- we need to scale them ourselves so
# they're on a similar numeric range to everything else. Neural nets are
# sensitive to feature scale; an unscaled 'Amount' of e.g. 5000 would
# dominate the loss compared to PCA features that hover around -3 to 3.
amount_scaler = StandardScaler()
time_scaler = StandardScaler()
df["Amount"] = amount_scaler.fit_transform(df[["Amount"]])
df["Time"] = time_scaler.fit_transform(df[["Time"]])
X = df.drop("Class", axis=1)
y = df["Class"]

# --- THE KEY STEP: split by class BEFORE train/test split ---
# We train ONLY on normal transactions. Fraud examples are held out
# entirely from training -- they only appear later, during evaluation.
X_normal = X[y == 0]
X_fraud = X[y == 1]

# Split normal data into train/validation (fraud stays untouched for now)
X_train, X_val = train_test_split(X_normal, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} normal transactions (fraud is NEVER seen during training)")
print(f"Validating on {len(X_val)} normal transactions")
print(f"Holding out {len(X_fraud)} fraud transactions for evaluation later")

# --- Build the autoencoder ---
input_dim = X_train.shape[1]  # 30 features

autoencoder = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    # ENCODER: compress 30 features down to 8 (the bottleneck)
    layers.Dense(20, activation="relu"),
    layers.Dense(14, activation="relu"),
    layers.Dense(8, activation="relu"),   # <-- bottleneck: forces compression
    # DECODER: reconstruct back up to 30 features
    layers.Dense(14, activation="relu"),
    layers.Dense(20, activation="relu"),
    layers.Dense(input_dim, activation="linear"),  # output = reconstructed input
])

autoencoder.compile(optimizer="adam", loss="mse")
# loss = "mse" (mean squared error) measures exactly what we care about:
# how far off is the reconstruction from the original input?

autoencoder.summary()

# --- Train ---
# Note: X_train is passed as BOTH the input AND the target.
# That's the defining trait of an autoencoder -- it's learning to predict
# itself, not some separate label.
history = autoencoder.fit(
    X_train, X_train,
    epochs=30,
    batch_size=256,
    validation_data=(X_val, X_val),
    shuffle=True,
    verbose=1,
)

# --- Save everything for the next script ---
os.makedirs(MODEL_DIR, exist_ok=True)
autoencoder.save(os.path.join(MODEL_DIR, "autoencoder.keras"))
joblib.dump(amount_scaler, os.path.join(MODEL_DIR, "amount_scaler.pkl"))
joblib.dump(time_scaler, os.path.join(MODEL_DIR, "time_scaler.pkl"))

print(f"\nModel saved to {MODEL_DIR}/autoencoder.keras")
print("Next: run 03_evaluate.py to see how well it catches fraud.")
