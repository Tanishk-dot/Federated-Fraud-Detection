# Federated Learning Training Results

## Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | paysim |
| FL Rounds | 20 |
| Clients | 10 |
| Clients/Round | 5 |
| Local Epochs | 1 |
| Batch Size | 128 |
| Learning Rate | 0.001 |
| FedProx μ | 0.5 |
| DP Enabled | False |
| Model Params | 1,069,388 |

## Final Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.7662** |
| Precision | 0.5934 |
| Recall | 0.9477 |
| **F1-Score** | **0.7298** |
| **ROC-AUC** | **0.9534** |
| PR-AUC | 0.9375 |
| FPR | 0.3246 |

## Confusion Matrix

|  | Predicted Legit | Predicted Fraud |
|--|-----------------|-----------------|
| **Actual Legit** | TN=5404 | FP=2597 |
| **Actual Fraud** | FN=209 | TP=3790 |

## Convergence Over Rounds

| Round | Accuracy | F1 | ROC-AUC | FPR |
|-------|----------|----|---------|-----|
| 1 | 0.8218 | 0.7578 | 0.9192 | 0.1855 |
| 2 | 0.7173 | 0.6738 | 0.9032 | 0.3620 |
| 3 | 0.6787 | 0.6563 | 0.9165 | 0.4423 |
| 4 | 0.6567 | 0.6371 | 0.8998 | 0.4668 |
| 5 | 0.6594 | 0.6423 | 0.9083 | 0.4697 |
| 6 | 0.6234 | 0.6201 | 0.9130 | 0.5259 |
| 7 | 0.5862 | 0.6046 | 0.9173 | 0.5954 |
| 8 | 0.6264 | 0.6264 | 0.9075 | 0.5302 |
| 9 | 0.6023 | 0.6188 | 0.9247 | 0.5808 |
| 10 | 0.6177 | 0.6261 | 0.9200 | 0.5537 |
| 11 | 0.6211 | 0.6266 | 0.8862 | 0.5453 |
| 12 | 0.5950 | 0.6113 | 0.8995 | 0.5852 |
| 13 | 0.7213 | 0.6881 | 0.9176 | 0.3791 |
| 14 | 0.6601 | 0.6492 | 0.9071 | 0.4817 |
| 15 | 0.6666 | 0.6545 | 0.8954 | 0.4738 |
| 16 | 0.6372 | 0.6372 | 0.8971 | 0.5221 |
| 17 | 0.5847 | 0.6071 | 0.8973 | 0.6043 |
| 18 | 0.6439 | 0.6425 | 0.9345 | 0.5142 |
| 19 | 0.6393 | 0.6421 | 0.9425 | 0.5263 |
| 20 | 0.7662 | 0.7298 | 0.9534 | 0.3246 |

## Plots

- ![Convergence](convergence_curves.png)
- ![Confusion Matrix](confusion_matrix.png)
- ![Precision & Recall](precision_recall_curve.png)
- ![FPR Over Rounds](fpr_over_rounds.png)
- ![Client Comparison](client_comparison.png)
