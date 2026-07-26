## Results

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Autoencoder | 0.9246 | 0.597 | 0.539 | 0.566 |
| Isolation Forest (threshold-tuned) | 0.9451 | 0.226 | 0.325 | 0.267 |

**Why one performed better than the other:**

Isolation Forest actually has a slightly better ROC-AUC (0.9451 vs 0.9246), meaning it's marginally better at *ranking* transactions from most to least suspicious in general. However, once both models are forced to commit to an actual threshold and make real fraud/not-fraud decisions, the autoencoder wins clearly — more than double the F1 score (0.566 vs 0.267). This suggests Isolation Forest's straight, one-feature-at-a-time splits struggle to draw a sharp, decisive boundary when fraud depends on *combinations* of features together, while the autoencoder — being a neural network — can learn these non-linear, multi-feature relationships and translate them into a more confident decision boundary.

**Recommendation:** the autoencoder, given its stronger precision/recall trade-off at a real operating threshold — which is what actually matters in a deployed fraud system, not just ranking ability.
