"""
The privacy-utility tradeoff curve: the standard evidence that "our DP
mechanism has a real, measurable effect" in any differential-privacy paper.

Trains the SAME federated (FedProx) model at a range of epsilon values, on
the SAME held-out test set, holding everything else fixed (rounds, clients,
local epochs, clip norm). This does NOT try to show DP "winning" on
accuracy - by construction, stronger privacy (lower epsilon) costs some
utility. What it shows is:
  (a) that the DP mechanism actually does something (accuracy visibly drops
      as epsilon shrinks - proof the noise isn't a no-op), and
  (b) exactly where the usable range starts for this model, so ε=100 (this
      project's default) is a stated, measured choice, not an arbitrary one.

Usage:
    python run_epsilon_sweep.py --epsilons 1 5 10 20 50 100 500 2000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_model_comparison import load_clients, run_federated  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="synthetic_paysim")
    parser.add_argument("--data-path", default=str(Path(__file__).resolve().parent.parent / "data"))
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent.parent / "results" / "comparison"))
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--clients-per-round", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--mu", type=float, default=0.1)
    parser.add_argument("--epsilons", type=float, nargs="+",
                         default=[1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0, 2000.0])
    parser.add_argument("--exclude-clients", type=str, default="",
                       help='Comma-separated client ids to exclude entirely '
                            '(e.g. "9" to drop the known paysim_real anomaly)')
    parser.add_argument("--max-samples-per-client", type=int, default=None,
                       help="Stratified cap on each client's TRAIN set, for runtime on "
                            "datasets much larger than the synthetic set this was sized "
                            "for. The shared test set is never capped.")
    parser.add_argument("--oversample-ratio", type=float, default=None,
                       help="Target per-batch fraud fraction (WeightedRandomSampler) - needed "
                            "at real PaySim-scale imbalance, where a plain shuffled DataLoader "
                            "collapses this architecture (verified). 0.02 is what "
                            "run_training.py found works best for this data.")
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    exclude = [int(c) for c in args.exclude_clients.split(",") if c.strip()]
    data_dir = Path(args.data_path) / args.dataset
    train, (test_X, test_y) = load_clients(
        data_dir, exclude_clients=exclude, max_samples_per_client=args.max_samples_per_client
    )
    num_features = test_X.shape[-1]
    train_fraud_rate = float(np.mean([y.numpy().mean() for _, y in train.values()]))
    print(f"Loaded {len(train)} clients. Held-out test set: {len(test_X)} sequences "
          f"({int(test_y.sum())} fraud) - same set used by run_model_comparison.py.")

    model_config = {
        "dataset": {"num_features": num_features},
        "model": {
            "hidden_dim": args.hidden_dim,
            "temporal": {"num_heads": 4, "num_layers": 2, "dropout": 0.1},
            "graph": {"num_layers": 2, "dropout": 0.1},
            "classifier": {"hidden_dim": 32, "dropout": 0.2},
            "contrastive": {"projection_dim": 32, "temperature": 0.07, "memory_bank_size": 1000},
        },
        "training": {
            "loss_weights": {"bce": 1.0, "contrastive": 0.0, "auxiliary": 0.0},
            "pos_weight": 8.0,
            "fraud_rate_prior": train_fraud_rate,
        },
    }

    points = []
    for eps in args.epsilons:
        np.random.seed(42)
        torch.manual_seed(42)
        t0 = time.time()
        m = run_federated(
            model_config, train, test_X, test_y,
            rounds=args.rounds, clients_per_round=args.clients_per_round,
            local_epochs=args.local_epochs, batch_size=args.batch_size,
            mu=args.mu, dp_epsilon=eps, label=f"eps={eps}",
            oversample_ratio=args.oversample_ratio,
        )
        points.append({
            "epsilon": eps,
            "accuracy": m["accuracy"], "precision": m["precision"],
            "recall": m["recall"], "f1": m["f1"], "roc_auc": m["roc_auc"],
            "train_seconds": round(time.time() - t0, 1),
        })
        print(f"epsilon={eps}: acc={m['accuracy']:.4f} f1={m['f1']:.4f} "
              f"rec={m['recall']:.4f} auc={m['roc_auc']:.4f}")

    # Also add the no-DP point (epsilon=None / infinity) as the ceiling.
    np.random.seed(42)
    torch.manual_seed(42)
    m = run_federated(
        model_config, train, test_X, test_y,
        rounds=args.rounds, clients_per_round=args.clients_per_round,
        local_epochs=args.local_epochs, batch_size=args.batch_size,
        mu=args.mu, dp_epsilon=None, label="no-DP",
        oversample_ratio=args.oversample_ratio,
    )
    points.insert(0, {
        "epsilon": None, "accuracy": m["accuracy"], "precision": m["precision"],
        "recall": m["recall"], "f1": m["f1"], "roc_auc": m["roc_auc"],
        "train_seconds": 0.0,
    })
    print(f"epsilon=None (no DP): acc={m['accuracy']:.4f} f1={m['f1']:.4f} "
          f"rec={m['recall']:.4f} auc={m['roc_auc']:.4f}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "dataset": args.dataset,
        "excluded_clients": exclude,
        "oversample_ratio": args.oversample_ratio,
        "test_set_size": len(test_X),
        "test_fraud_rate": float(test_y.float().mean()),
        "methodology": (
            "Same FedProx-federated model (mu={mu}), same {rounds} rounds, same "
            "held-out test set as run_model_comparison.py's baseline D, swept "
            "across epsilon with everything else held fixed. epsilon=null means "
            "no DP noise (the ceiling)."
        ).format(mu=args.mu, rounds=args.rounds),
        "points": points,
    }
    with open(out_dir / "epsilon_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_dir / 'epsilon_sweep.json'}")


if __name__ == "__main__":
    main()
