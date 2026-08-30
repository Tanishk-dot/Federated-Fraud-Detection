"""
Generate a synthetic-but-realistic PaySim-style federated dataset.

The real PaySim dataset used originally (`datasets_capstone/preprocessed_datasets_final/`)
is not present on this machine (it lived on an external drive during
development). This script produces a stand-in dataset with the *same* schema
FraudDetectionDataset expects, built from the fraud/legit transaction
archetypes documented in docs/TESTING_GUIDE.md, using the single feature
encoding in src/data/feature_engineering.py (the same one the dashboard's
"Test Your Data" tab uses at inference time).

v2 design note (see docs/ARCHITECTURE.md "Model comparison"): the first
version of this generator drew fraud and legit transactions from *disjoint*
value ranges (e.g. legit amount capped at $400, every fraud archetype above
$400 or below $3) - a threshold on a single feature separated the classes
almost perfectly, so every model (down to a plain RandomForest) scored
~100% on every metric. That's not a fabricated number, but it wasn't a
believable demo either - real fraud and legit transactions overlap heavily
in amount, distance, timing, which is exactly what makes fraud detection
hard and is exactly what would show daylight between models. v2 instead
draws every archetype from the *same distribution families* as legit,
shifted in mean/variance rather than clipped to a separate range, so the
classes overlap substantially, and folds in a fraction of "quiet fraud"
(drawn straight from the legit distribution, still labeled fraud) so no
feature-based model can reach 100% recall by construction - some fraud
genuinely looks identical to a legit transaction on these 8 fields, same as
in reality.

It's still rule-based synthetic data, not a substitute for the real 2.86M-row
PaySim set - treat metrics trained on it as a pipeline correctness /
relative-comparison check, not a claim about real-world performance. Swap in
real data by pointing configs/experiment_config.yaml's dataset.data_path at a
directory with the same client_i/{split}_temporal.pt layout.

Usage:
    python generate_synthetic_paysim.py --out ../data/synthetic_paysim --clients 10
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.feature_engineering import build_sequence, SEQ_LEN, NUM_FEATURES

ALL_MERCHANTS = ["Grocery", "Gas Station", "Pharmacy", "Restaurant", "ATM",
                  "Retail", "Electronics", "Online", "Travel", "Jewelry"]

RNG_SEED = 42

# Fraction of "fraud"-labeled transactions that are drawn straight from the
# legit distribution (behaviorally indistinguishable on these 8 features).
# Real fraud has this too - not everything that ends up charged back looks
# anomalous at the transaction-feature level. This puts a hard ceiling on
# achievable recall/accuracy well under 100%, by construction.
QUIET_FRAUD_FRAC = 0.12

# Multiplicative jitter applied to every numeric feature after generation,
# so archetype boundaries aren't perfectly crisp even within a class.
JITTER_SIGMA = 0.15


def _jitter(x, rng, lo, hi):
    x = x * float(rng.normal(1.0, JITTER_SIGMA))
    return float(np.clip(x, lo, hi))


def _legit_txn(rng):
    hour_weights = np.array([1, 1, 1, 1, 1, 2, 4, 6, 7, 7, 7, 7,
                              7, 7, 7, 7, 7, 7, 6, 5, 4, 3, 2, 1], dtype=float)
    hour_weights /= hour_weights.sum()
    merchant_p = np.array([0.16, 0.14, 0.10, 0.14, 0.10, 0.14, 0.08, 0.09, 0.03, 0.02])
    merchant_p /= merchant_p.sum()
    return {
        "amount": _jitter(np.clip(rng.lognormal(mean=3.8, sigma=1.1), 1, 3000), rng, 1, 3000),
        "hour": int(rng.choice(24, p=hour_weights)),
        "day_of_week": int(rng.integers(0, 7)),
        "merchant_type": str(rng.choice(ALL_MERCHANTS, p=merchant_p)),
        "distance_km": _jitter(np.clip(rng.exponential(20.0), 0, 500), rng, 0, 500),
        "minutes_since_last": _jitter(np.clip(rng.exponential(180.0), 1, 1440), rng, 1, 1440),
        "card_age_days": float(rng.integers(30, 2000)),
        "txns_last_24h": int(np.clip(rng.poisson(2.5), 0, 15)),
    }


# Fraud archetypes drawn from docs/TESTING_GUIDE.md so labels stay consistent
# with what the dashboard documents as "should be flagged" - but now shifted
# distributions instead of disjoint ranges, so they overlap with legit.
def _fraud_txn(rng):
    if rng.random() < QUIET_FRAUD_FRAC:
        return _legit_txn(rng)

    archetype = rng.choice([
        "impossible_velocity", "high_value_night", "card_testing",
        "travel_fraud", "burst_online",
    ])
    night_weights = np.array([6, 6, 6, 6, 5, 4, 2, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6], dtype=float)
    night_weights /= night_weights.sum()

    if archetype == "impossible_velocity":
        return {
            "amount": _jitter(np.clip(rng.lognormal(mean=5.0, sigma=1.0), 1, 3000), rng, 1, 3000),
            "hour": int(rng.choice(24, p=night_weights)),
            "day_of_week": int(rng.integers(0, 7)),
            "merchant_type": str(rng.choice(["Online", "Travel", "Electronics", "Jewelry", "Retail"])),
            "distance_km": _jitter(np.clip(rng.exponential(250.0), 0, 1500), rng, 0, 1500),
            "minutes_since_last": _jitter(np.clip(rng.exponential(25.0), 1, 1440), rng, 1, 1440),
            "card_age_days": float(rng.integers(30, 2000)),
            "txns_last_24h": int(np.clip(rng.poisson(4), 0, 15)),
        }
    if archetype == "high_value_night":
        card_age = rng.uniform(1, 60) if rng.random() < 0.6 else rng.uniform(60, 2000)
        return {
            "amount": _jitter(np.clip(rng.lognormal(mean=6.5, sigma=1.3), 1, 9500), rng, 1, 9500),
            "hour": int(rng.choice(24, p=night_weights)),
            "day_of_week": int(rng.integers(0, 7)),
            "merchant_type": str(rng.choice(["Online", "Electronics", "Jewelry", "Retail"], p=[0.5, 0.2, 0.2, 0.1])),
            "distance_km": _jitter(np.clip(rng.exponential(15.0), 0, 500), rng, 0, 500),
            "minutes_since_last": _jitter(np.clip(rng.exponential(150.0), 1, 1440), rng, 1, 1440),
            "card_age_days": float(card_age),
            "txns_last_24h": int(np.clip(rng.poisson(2), 0, 15)),
        }
    if archetype == "card_testing":
        return {
            "amount": _jitter(np.clip(rng.exponential(3.0), 0.5, 50), rng, 0.5, 50),
            "hour": int(rng.choice(24, p=night_weights)),
            "day_of_week": int(rng.integers(0, 7)),
            "merchant_type": str(rng.choice(["Online", "Retail", "Electronics"], p=[0.7, 0.2, 0.1])),
            "distance_km": _jitter(np.clip(rng.exponential(200.0), 0, 1500), rng, 0, 1500),
            "minutes_since_last": _jitter(np.clip(rng.exponential(8.0), 0.5, 1440), rng, 0.5, 1440),
            "card_age_days": float(rng.uniform(1, 200)),
            "txns_last_24h": int(np.clip(rng.poisson(10), 0, 30)),
        }
    if archetype == "travel_fraud":
        return {
            "amount": _jitter(np.clip(rng.lognormal(mean=5.5, sigma=1.0), 1, 5000), rng, 1, 5000),
            "hour": int(rng.choice(24, p=night_weights)),
            "day_of_week": int(rng.integers(0, 7)),
            "merchant_type": str(rng.choice(["Travel", "Online", "Jewelry"], p=[0.6, 0.25, 0.15])),
            "distance_km": _jitter(np.clip(rng.exponential(400.0), 0, 2000), rng, 0, 2000),
            "minutes_since_last": _jitter(np.clip(rng.exponential(20.0), 1, 1440), rng, 1, 1440),
            "card_age_days": float(rng.uniform(1, 500)),
            "txns_last_24h": int(np.clip(rng.poisson(2), 0, 15)),
        }
    # burst_online
    return {
        "amount": _jitter(np.clip(rng.exponential(15.0), 1, 200), rng, 1, 200),
        "hour": int(rng.choice(24, p=night_weights)),
        "day_of_week": int(rng.integers(0, 7)),
        "merchant_type": str(rng.choice(["Online", "Retail"], p=[0.85, 0.15])),
        "distance_km": _jitter(np.clip(rng.exponential(150.0), 0, 1200), rng, 0, 1200),
        "minutes_since_last": _jitter(np.clip(rng.exponential(6.0), 0.5, 1440), rng, 0.5, 1440),
        "card_age_days": float(rng.uniform(1, 200)),
        "txns_last_24h": int(np.clip(rng.poisson(12), 0, 30)),
    }


def generate_client_split(rng, num_samples, fraud_rate):
    sequences = np.zeros((num_samples, SEQ_LEN, NUM_FEATURES), dtype=np.float32)
    labels = np.zeros(num_samples, dtype=np.int64)

    num_fraud = int(num_samples * fraud_rate)
    fraud_idx = set(rng.choice(num_samples, size=num_fraud, replace=False).tolist())

    for i in range(num_samples):
        is_fraud = i in fraud_idx
        history = [_legit_txn(rng) for _ in range(SEQ_LEN - 1)]
        history.append(_fraud_txn(rng) if is_fraud else _legit_txn(rng))
        sequences[i] = build_sequence(history, seq_len=SEQ_LEN)
        labels[i] = 1 if is_fraud else 0

    return sequences, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="../data/synthetic_paysim")
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--samples-per-client", type=int, default=4000)
    parser.add_argument("--fraud-rate", type=float, default=0.06)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {"train": 0.7, "val": 0.15, "test": 0.15}

    for client_id in range(args.clients):
        client_rng = np.random.default_rng(RNG_SEED + client_id)
        client_dir = out_dir / f"client_{client_id}"
        client_dir.mkdir(exist_ok=True)

        n = args.samples_per_client
        sequences, labels = generate_client_split(client_rng, n, args.fraud_rate)
        perm = client_rng.permutation(n)
        sequences, labels = sequences[perm], labels[perm]

        n_train = int(n * splits["train"])
        n_val = int(n * splits["val"])
        idx = {"train": (0, n_train), "val": (n_train, n_train + n_val), "test": (n_train + n_val, n)}

        for split, (lo, hi) in idx.items():
            torch.save({
                "sequences": torch.tensor(sequences[lo:hi]),
                "labels": torch.tensor(labels[lo:hi]),
            }, client_dir / f"{split}_temporal.pt")

        print(f"client_{client_id}: {n} samples, "
              f"fraud_rate={labels.mean():.3f} -> train={n_train} val={n_val} test={n-n_train-n_val}")

    metadata_dir = out_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    with open(metadata_dir / "dataset_info.json", "w") as f:
        json.dump({
            "name": "synthetic_paysim",
            "note": "Rule-based synthetic data (v2: overlapping fraud/legit distributions "
                     f"+ {QUIET_FRAUD_FRAC:.0%} 'quiet fraud' drawn from the legit distribution, "
                     "so no feature-based model can reach 100% by construction). Real PaySim "
                     "data unavailable. See docs/ARCHITECTURE.md.",
            "num_clients": args.clients,
            "samples_per_client": args.samples_per_client,
            "fraud_rate": args.fraud_rate,
            "quiet_fraud_frac": QUIET_FRAUD_FRAC,
            "seq_len": SEQ_LEN,
            "num_features": NUM_FEATURES,
        }, f, indent=2)

    print(f"\nDone. Wrote {args.clients} clients to {out_dir}")


if __name__ == "__main__":
    main()
