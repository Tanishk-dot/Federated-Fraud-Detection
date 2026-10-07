"""
Quick, no-retraining check: is 0.5 the right decision threshold for the
real paysim_real checkpoint, or is the model ranking fraud well (high
ROC-AUC) while 0.5 throws away recall?

Loads the saved results/paysim_real/global_model.pt and the same pooled
test set run_training.py builds (all clients' test splits, client_9
excluded by default to stay representative), scores every sequence once,
and reports precision/recall/F1 across a grid of thresholds plus the
threshold that maximizes F1.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_training import load_and_fix_client_data
from src.models.temporal_graph_transformer import TemporalGraphTransformer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="results/paysim_real/global_model.pt")
    parser.add_argument("--data-path", default="data")
    parser.add_argument("--dataset", default="paysim_real")
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--exclude-clients", default="9")
    parser.add_argument("--max-samples", type=int, default=50000)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = TemporalGraphTransformer(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    excluded = {int(c) for c in args.exclude_clients.split(",") if c.strip() != ""}
    data_path = Path(args.data_path) / args.dataset

    test_X, test_y = [], []
    for cid in range(args.clients):
        if cid in excluded:
            continue
        d = load_and_fix_client_data(data_path, cid, max_samples_per_client=args.max_samples)
        tx, ty = d["test"]
        test_X.append(tx)
        test_y.append(ty)
    X = torch.cat(test_X, dim=0)
    y = torch.cat(test_y, dim=0).numpy()

    probs = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            batch = X[i:i + 512]
            out = model(batch, graph_features=None, labels=None)
            probs.append(out["fraud_prob"].numpy())
    probs = np.concatenate(probs)

    print(f"\nTest set: {len(y)} sequences, {int(y.sum())} fraud ({y.mean()*100:.3f}%)")
    print(f"Prob range: [{probs.min():.4f}, {probs.max():.4f}], mean={probs.mean():.4f}")

    print(f"\n{'threshold':>10} {'precision':>10} {'recall':>8} {'f1':>8} {'tp':>6} {'fp':>6} {'fn':>6}")
    for t in [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]:
        preds = (probs > t).astype(int)
        tp = int(((preds == 1) & (y == 1)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        fn = int(((preds == 0) & (y == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-10)
        print(f"{t:>10.2f} {prec:>10.4f} {rec:>8.4f} {f1:>8.4f} {tp:>6} {fp:>6} {fn:>6}")

    precisions, recalls, thresholds = precision_recall_curve(y, probs)
    f1s = 2 * precisions * recalls / np.clip(precisions + recalls, 1e-10, None)
    best_idx = np.argmax(f1s[:-1])  # last point has no corresponding threshold
    print(f"\nBest-F1 threshold: {thresholds[best_idx]:.4f} "
          f"-> precision={precisions[best_idx]:.4f}, recall={recalls[best_idx]:.4f}, "
          f"f1={f1s[best_idx]:.4f}")


if __name__ == "__main__":
    main()
