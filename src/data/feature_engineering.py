"""
Shared feature engineering for the fraud detection model.

This is the single source of truth for how a raw transaction (the kind a
human or a CSV upload would supply) becomes the 10-dim feature vector the
model was trained on, and how a rolling window of transactions becomes the
(seq_len, 10) sequence the model expects as input.

Both training-data generation (`generate_synthetic_dataset`) and live/batch
inference (dashboard "Test Your Data" tab) import from here, specifically so
they can't drift apart — a value computed for training means the same thing
at inference time.
"""

import numpy as np
import torch

SEQ_LEN = 10
NUM_FEATURES = 10

# Raw columns a human (or a CSV) provides for one transaction.
RAW_COLUMNS = [
    "amount",              # transaction amount in dollars
    "hour",                # 0-23
    "day_of_week",         # 0=Monday .. 6=Sunday
    "merchant_type",       # one of MERCHANT_RISK.keys()
    "distance_km",         # distance from the account's previous transaction
    "minutes_since_last",  # minutes since the account's previous transaction
    "card_age_days",       # age of the card/account in days
    "txns_last_24h",       # number of transactions by this account in the last 24h
]

# Merchant risk tiers — matches docs/TESTING_GUIDE.md
MERCHANT_RISK = {
    "Grocery": 0.0, "Gas Station": 0.0, "Pharmacy": 0.0,
    "Restaurant": 0.35, "ATM": 0.35, "Retail": 0.35,
    "Electronics": 0.7,
    "Online": 1.0, "Travel": 1.0, "Jewelry": 1.0,
}

# Defaults used when a caller (e.g. a CSV upload's column mapping) doesn't
# have a source column for a required field - a neutral, "unremarkable
# transaction" value, so an unmapped field pulls the score toward "normal"
# rather than silently toward "fraud" or crashing. The API layer that builds
# the mapping UI should show these so the approximation is visible, not hidden.
DEFAULTS = {
    "amount": 60.0,
    "hour": 14.0,
    "day_of_week": 2.0,
    "merchant_type": "Retail",
    "distance_km": 5.0,
    "minutes_since_last": 240.0,
    "card_age_days": 365.0,
    "txns_last_24h": 2.0,
}

# Normalization caps — values are clipped to [0, cap] then divided by cap.
# Chosen generously above typical legitimate values so fraud outliers still
# read as "close to 1" rather than being clipped into indistinguishability.
_CAPS = {
    "amount": 10_000.0,
    "distance_km": 1500.0,
    "minutes_since_last": 1440.0,   # 24h
    "card_age_days": 1095.0,        # 3 years
    "txns_last_24h": 30.0,
    "velocity_kmh": 1500.0,
}


def _norm(value, cap):
    return float(np.clip(value, 0.0, cap) / cap)


def transaction_to_features(txn: dict) -> np.ndarray:
    """
    Convert one raw transaction (dict with RAW_COLUMNS keys) into the
    10-dim feature vector the model consumes for a single sequence step.

    Feature order (must match training):
      0 amount_norm         5 time_since_last_norm
      1 hour_norm           6 velocity_kmh_norm   (derived: distance/time)
      2 day_of_week_norm    7 card_age_norm
      3 merchant_risk       8 txns_24h_norm
      4 distance_norm       9 is_weekend
    """
    amount = float(txn.get("amount", DEFAULTS["amount"]))
    hour = float(txn.get("hour", DEFAULTS["hour"]))
    dow = float(txn.get("day_of_week", DEFAULTS["day_of_week"]))
    merchant = txn.get("merchant_type", DEFAULTS["merchant_type"])
    distance = float(txn.get("distance_km", DEFAULTS["distance_km"]))
    minutes = float(txn.get("minutes_since_last", DEFAULTS["minutes_since_last"]))
    card_age = float(txn.get("card_age_days", DEFAULTS["card_age_days"]))
    txns_24h = float(txn.get("txns_last_24h", DEFAULTS["txns_last_24h"]))

    merchant_risk = MERCHANT_RISK.get(merchant, 0.5)  # unknown merchant -> medium risk

    # Implied velocity in km/h; guard against div-by-zero for back-to-back txns.
    hours_elapsed = max(minutes, 1.0) / 60.0
    velocity_kmh = distance / hours_elapsed

    is_weekend = 1.0 if dow >= 5 else 0.0

    return np.array([
        _norm(amount, _CAPS["amount"]),
        np.clip(hour, 0, 23) / 23.0,
        np.clip(dow, 0, 6) / 6.0,
        merchant_risk,
        _norm(distance, _CAPS["distance_km"]),
        _norm(minutes, _CAPS["minutes_since_last"]),
        _norm(velocity_kmh, _CAPS["velocity_kmh"]),
        _norm(card_age, _CAPS["card_age_days"]),
        _norm(txns_24h, _CAPS["txns_last_24h"]),
        is_weekend,
    ], dtype=np.float32)


def build_sequence(history: list, seq_len: int = SEQ_LEN) -> np.ndarray:
    """
    Turn an account's transaction history (list of raw dicts, oldest first,
    most recent last) into a (seq_len, NUM_FEATURES) array.

    If there are fewer than `seq_len` transactions, the earliest one is
    repeated to left-pad the window — a documented simplifying assumption
    for accounts/uploads with limited history (including a single
    transaction with no history at all).
    """
    if len(history) == 0:
        raise ValueError("history must contain at least one transaction")

    feats = [transaction_to_features(t) for t in history[-seq_len:]]
    while len(feats) < seq_len:
        feats.insert(0, feats[0])
    return np.stack(feats, axis=0)


def sequence_to_tensor(seq: np.ndarray) -> torch.Tensor:
    """(seq_len, NUM_FEATURES) numpy -> (1, seq_len, NUM_FEATURES) float32 tensor."""
    return torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
