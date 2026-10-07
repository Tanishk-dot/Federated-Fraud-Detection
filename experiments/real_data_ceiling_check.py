"""
Ceiling check: how well can this exact architecture + exact real PaySim
features do with ALL federation/DP noise removed - one centralized model,
all 9 clients' real train data pooled directly (no FedProx proximal term,
no DP noise, no per-client partitioning)?

This isolates "is 90s F1/recall even reachable with this feature set" from
"is the federated training loop costing us accuracy" - if even this
ceiling falls well short of 90s, no amount of federated-side tuning will
get there either, and that needs to be said plainly rather than chased
indefinitely.

Usage:
    python experiments/real_data_ceiling_check.py --epochs 10 --oversample-ratio 0.02
"""
import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_training import DEFAULT_CONFIG, evaluate_model  # noqa: E402
from src.models.temporal_graph_transformer import TemporalGraphTransformer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/paysim_real")
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--exclude-clients", default="9")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--pos-weight", type=float, default=8.0)
    parser.add_argument("--fraud-rate-prior", type=float, default=0.001)
    parser.add_argument("--oversample-ratio", type=float, default=0.0,
                         help="0 disables oversampling; 0.02 matches the tuned federated fix")
    parser.add_argument("--hidden-dim", type=int, default=32,
                         help="32 matches the DP-constrained federated model; try 64 to see "
                              "if capacity (not federation) was the limit, since this run has no DP")
    parser.add_argument("--max-train-samples", type=int, default=None,
                         help="cap pooled centralized train set size (stratified) for a faster run")
    args = parser.parse_args()

    excluded = {int(c) for c in args.exclude_clients.split(",") if c.strip()}
    data_path = Path(args.data_path)

    train_X, train_y, test_X, test_y = [], [], [], []
    for cid in range(args.clients):
        if cid in excluded:
            continue
        cdir = data_path / f"client_{cid}"
        tr = torch.load(cdir / "train_temporal.pt", map_location="cpu", weights_only=False)
        te = torch.load(cdir / "test_temporal.pt", map_location="cpu", weights_only=False)
        train_X.append(tr["sequences"]); train_y.append(tr["labels"])
        test_X.append(te["sequences"]); test_y.append(te["labels"])

    X_train = torch.cat(train_X); y_train = torch.cat(train_y)
    X_test = torch.cat(test_X); y_test = torch.cat(test_y)

    if args.max_train_samples is not None and len(X_train) > args.max_train_samples:
        idx_pos = (y_train == 1).nonzero(as_tuple=True)[0]
        idx_neg = (y_train == 0).nonzero(as_tuple=True)[0]
        frac = args.max_train_samples / len(X_train)
        n_pos = max(int(len(idx_pos) * frac), 1)
        n_neg = args.max_train_samples - n_pos
        sel_pos = idx_pos[torch.randperm(len(idx_pos))[:n_pos]]
        sel_neg = idx_neg[torch.randperm(len(idx_neg))[:n_neg]]
        sel = torch.cat([sel_pos, sel_neg])
        sel = sel[torch.randperm(len(sel))]
        X_train, y_train = X_train[sel], y_train[sel]

    print(f"Pooled centralized train: {len(X_train):,} seqs, {int(y_train.sum())} fraud "
          f"({y_train.float().mean()*100:.3f}%)")
    print(f"Pooled test: {len(X_test):,} seqs, {int(y_test.sum())} fraud "
          f"({y_test.float().mean()*100:.3f}%)")

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["dataset"]["num_features"] = X_train.shape[-1]
    config["model"]["hidden_dim"] = args.hidden_dim
    config["model"]["classifier"]["hidden_dim"] = args.hidden_dim
    config["training"]["pos_weight"] = args.pos_weight
    config["training"]["fraud_rate_prior"] = args.fraud_rate_prior

    if args.oversample_ratio > 0:
        n_pos = int(y_train.sum().item()); n_neg = len(y_train) - n_pos
        weight_per_class = torch.tensor([
            (1.0 - args.oversample_ratio) / n_neg,
            args.oversample_ratio / n_pos,
        ])
        sample_weights = weight_per_class[y_train.long()]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(y_train), replacement=True)
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256)

    print(f"Model params: {sum(p.numel() for p in TemporalGraphTransformer(config).parameters()):,} "
          f"(hidden_dim={args.hidden_dim})")

    max_retries = 3
    best_metrics, best_f1 = None, -1.0
    t0 = time.time()
    for attempt in range(max_retries):
        model = TemporalGraphTransformer(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        attempt_best, attempt_best_f1 = None, -1.0

        for epoch in range(args.epochs):
            model.train()
            epoch_loss = 0.0
            for sequences, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(sequences, graph_features=None, labels=labels)
                loss, _ = model.compute_loss(outputs, labels)
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            m = evaluate_model(model, test_loader, config, "cpu")
            collapsed = m["recall"] == 0.0 and abs(m["roc_auc"] - 0.5) < 1e-6
            print(f"attempt {attempt+1}/{max_retries} epoch {epoch+1}/{args.epochs}: "
                  f"loss={epoch_loss/len(train_loader):.4f} prec={m['precision']:.4f} "
                  f"rec={m['recall']:.4f} f1={m['f1']:.4f} auc={m['roc_auc']:.4f}"
                  f" [{time.time()-t0:.0f}s]" + (" [collapsed]" if collapsed else ""))
            if not collapsed and m["f1"] > attempt_best_f1:
                attempt_best, attempt_best_f1 = m, m["f1"]

        if attempt_best_f1 > best_f1:
            best_metrics, best_f1 = attempt_best, attempt_best_f1
        if best_f1 > 0:
            break
        print(f"  attempt {attempt+1}: every epoch collapsed - retrying with a fresh init...")

    if best_metrics is None:
        print("\nEvery attempt collapsed to a degenerate constant prediction - "
              "reporting that honestly rather than hiding it.")
    else:
        print(f"\nBest epoch: acc={best_metrics['accuracy']:.4f} prec={best_metrics['precision']:.4f} "
              f"rec={best_metrics['recall']:.4f} f1={best_metrics['f1']:.4f} auc={best_metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
