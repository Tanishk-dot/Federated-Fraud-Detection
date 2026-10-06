"""
Rebuilds correct, usable train/val/test splits for the real PaySim dataset
under preprocessed_datasets_csv/paysim/.

Why this exists: the CSVs as delivered have val/test splits that are 100%
fraud (zero legitimate sequences) for every client - verified by reading the
full label distribution of every client_i_{val,test}.csv, not a sample. A
split with only one class present makes precision, false-positive rate, and
ROC-AUC undefined, so training against the given splits directly would
produce meaningless evaluation numbers. configs/paysim.yaml, referenced by
this dataset's own metadata.json as the file that built it, does not exist
anywhere in this repo, so the original split logic can't be inspected or
repaired in place.

A first fix attempt (pool train+val+test, then re-split) was tried and
rejected: pooling just redistributes the same corruption evenly, producing
a ~30% fraud rate in every split for every client - nowhere near real
PaySim's ~0.13% - which shows the val/test files aren't misassigned
legitimate rows, they're a separate, massively fraud-inflated pool on top
of a clean train file.

Actual fix: per client, use ONLY that client's own train CSV (fraud rate
there is 0.06%-0.17% for 9 of 10 clients - the right order of magnitude for
real PaySim's 0.1276%) and rebuild a fresh stratified 70/15/15 split from
that alone. The original val/test CSVs are discarded entirely, not reused
in any form. Client_id boundaries are preserved (each client keeps exactly
its own train rows).

Client 9 is a known anomaly (its own train file alone is ~46.9% fraud, vs
<0.2% for every other client) and is NOT corrected here - it is rebuilt
from its own train pool like every other client, so its elevated fraud rate
persists by design and must be reported, not hidden.

Usage:
    python build_real_paysim_splits.py \
        --src ../preprocessed_datasets_csv/paysim \
        --out ../data/paysim_real
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

FEATURE_COLS = [f"feature_{i}" for i in range(10)]
SEQ_LEN = 10
VAL_FRAC = 0.15
TEST_FRAC = 0.15
SEED = 42


def load_split(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path, usecols=["sequence_id", "timestep", "label"] + FEATURE_COLS)
    df = df.sort_values(["sequence_id", "timestep"], kind="mergesort")

    seq_ids = df["sequence_id"].unique()
    seq_ids.sort()
    n_seq = len(seq_ids)
    if not (seq_ids[0] == 0 and seq_ids[-1] == n_seq - 1):
        raise ValueError(f"{csv_path.name}: sequence_id is not a dense 0..{n_seq-1} range")

    counts = df.groupby("sequence_id", sort=True).size()
    if not (counts == SEQ_LEN).all():
        bad = counts[counts != SEQ_LEN]
        raise ValueError(f"{csv_path.name}: {len(bad)} sequences don't have exactly {SEQ_LEN} rows")

    features = df[FEATURE_COLS].to_numpy(dtype=np.float32).reshape(n_seq, SEQ_LEN, len(FEATURE_COLS))
    labels = df.groupby("sequence_id", sort=True)["label"].first().to_numpy(dtype=np.int64)
    return features, labels


def stratified_three_way(labels: np.ndarray, val_frac: float, test_frac: float, seed: int):
    idx = np.arange(len(labels))
    train_idx, rest_idx = train_test_split(
        idx, test_size=val_frac + test_frac, stratify=labels, random_state=seed
    )
    rest_labels = labels[rest_idx]
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=test_frac / (val_frac + test_frac), stratify=rest_labels, random_state=seed
    )
    return train_idx, val_idx, test_idx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=str, default="../preprocessed_datasets_csv/paysim")
    parser.add_argument("--out", type=str, default="../data/paysim_real")
    parser.add_argument("--clients", type=int, default=10)
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    report = {}

    for cid in range(args.clients):
        print(f"\n=== client_{cid} ===")
        orig_counts = {}
        for split in ("train", "val", "test"):
            csv_path = src / f"client_{cid}_{split}.csv"
            _, labels = load_split(csv_path)
            n_fraud = int(labels.sum())
            orig_counts[split] = {"n": len(labels), "fraud": n_fraud, "fraud_rate": n_fraud / len(labels)}
            print(f"  original {split}: {len(labels):>7,} seqs, {n_fraud:>6} fraud "
                  f"({n_fraud/len(labels)*100:.3f}%)")

        # only the train CSV is trustworthy (see module docstring) - val/test discarded
        all_feats, all_labels = load_split(src / f"client_{cid}_train.csv")

        pooled_n = len(all_labels)
        pooled_fraud = int(all_labels.sum())
        print(f"  using train-only pool: {pooled_n:>7,} seqs, {pooled_fraud:>6} fraud "
              f"({pooled_fraud/pooled_n*100:.3f}%)")

        train_idx, val_idx, test_idx = stratified_three_way(all_labels, VAL_FRAC, TEST_FRAC, SEED)

        client_dir = out / f"client_{cid}"
        client_dir.mkdir(exist_ok=True)
        rebuilt_counts = {}
        for split, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
            feats = all_feats[idx]
            labels = all_labels[idx]
            torch.save(
                {"sequences": torch.from_numpy(feats), "labels": torch.from_numpy(labels)},
                client_dir / f"{split}_temporal.pt",
            )
            n_fraud = int(labels.sum())
            rebuilt_counts[split] = {"n": len(labels), "fraud": n_fraud, "fraud_rate": n_fraud / len(labels)}
            print(f"  rebuilt  {split}: {len(labels):>7,} seqs, {n_fraud:>6} fraud "
                  f"({n_fraud/len(labels)*100:.3f}%)")

        report[f"client_{cid}"] = {
            "original": orig_counts,
            "train_only_pool": {"n": pooled_n, "fraud": pooled_fraud, "fraud_rate": pooled_fraud / pooled_n},
            "rebuilt": rebuilt_counts,
        }

    overall_pool_fraud_rate = {
        cid: report[cid]["train_only_pool"]["fraud_rate"] for cid in report
    }
    anomalous = {
        cid: rate for cid, rate in overall_pool_fraud_rate.items()
        if rate > 0.05
    }

    metadata = {
        "name": "paysim_real",
        "note": (
            "Real PaySim transactions (preprocessed_datasets_csv/paysim/), re-split by "
            "this project. The delivered val/test CSVs were discarded: every client's "
            "val and test file was 100% fraud (zero legitimate sequences) - unusable "
            "for precision/ROC-AUC/confusion-matrix evaluation. A first fix attempt "
            "(pool train+val+test, re-split) was also rejected: it just redistributed "
            "the same corruption into a uniform ~30% fraud rate per client, far from "
            "real PaySim's ~0.13% - proof the val/test files are a separate inflated "
            "fraud pool, not misassigned legitimate rows. Final fix: rebuilt all three "
            "splits from each client's TRAIN CSV alone (0.06%-0.17% fraud for 9/10 "
            "clients - the right order of magnitude for real PaySim), via a fresh "
            "stratified 70/15/15 split. No rows were fabricated - every sequence here "
            "came from the original train CSVs; client_id boundaries are preserved."
        ),
        "known_anomaly": (
            f"client(s) {list(anomalous.keys())} have an abnormally high fraud rate "
            f"in their OWN train file ({ {k: f'{v*100:.2f}%' for k, v in anomalous.items()} }) vs "
            "<0.2% for the rest - present in the original data, root cause unknown since "
            "configs/paysim.yaml (referenced by the source metadata.json) does not "
            "exist in this repo. NOT corrected here - reported as-is."
        ) if anomalous else "none detected",
        "num_clients": args.clients,
        "seq_len": SEQ_LEN,
        "num_features": len(FEATURE_COLS),
        "val_frac": VAL_FRAC,
        "test_frac": TEST_FRAC,
        "seed": SEED,
        "per_client": report,
    }
    meta_dir = out / "metadata"
    meta_dir.mkdir(exist_ok=True)
    with open(meta_dir / "dataset_info.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved rebuilt splits to {out}")
    if anomalous:
        print(f"ANOMALY: {anomalous} - see metadata/dataset_info.json 'known_anomaly'")


if __name__ == "__main__":
    main()
