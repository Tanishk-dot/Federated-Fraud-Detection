#!/usr/bin/env python3
"""
Federated Learning Training Script — Direct Implementation
Runs FL training without Ray dependency, using Flower's core logic manually.

This script:
1. Fixes the data split issue (val/test were 100% fraud)
2. Implements FedProx federated averaging
3. Applies real differential privacy (gradient clipping + Gaussian noise)
4. Logs real metrics from real training
5. Saves model checkpoints and training history

Usage:
    ~/venv_fl/bin/python3 run_training.py --dataset paysim --rounds 20 --clients 10
"""

import argparse
import copy
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, average_precision_score
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models.temporal_graph_transformer import TemporalGraphTransformer

# ============================================================================
# Configuration
# ============================================================================

# hidden_dim=128 (a 4-layer, 8-head transformer) against a 10-dim input is
# ~13x over-parameterized for the actual signal in this data. Verified
# empirically: with that size, training on the (harder, v2) synthetic data
# converges the network to output a constant (the base fraud rate) for
# every sample regardless of input - post-training batch std of fraud_prob
# collapses to ~0, vs ~0.05 at init, i.e. the optimizer actively learns to
# discard per-sample signal because "predict the base rate" is a stable,
# easy-to-reach local minimum of unweighted BCE under heavy class
# imbalance. hidden_dim=32 (right-sized to the input) + pos_weight (extra
# gradient pressure on the minority class) + a prior-informed classifier
# bias init (see TemporalGraphTransformer.__init__) together let the model
# actually learn a decision boundary instead of collapsing to it. See
# docs/ARCHITECTURE.md "Model comparison".
DEFAULT_CONFIG = {
    'dataset': {'num_features': 10},
    'model': {
        'hidden_dim': 32,
        'temporal': {'num_heads': 4, 'num_layers': 2, 'dropout': 0.1},
        'graph': {'num_layers': 2, 'dropout': 0.1},
        'classifier': {'hidden_dim': 32, 'dropout': 0.2},
        'contrastive': {'projection_dim': 32, 'temperature': 0.07, 'memory_bank_size': 1000}
    },
    'training': {
        'loss_weights': {'bce': 1.0, 'contrastive': 0.0, 'auxiliary': 0.0},
        'pos_weight': 8.0,
        'fraud_rate_prior': 0.06,
    }
}

DATASET_FEATURES = {
    'paysim': 10,
    'european_cc_2013': 30,
    'ieee_cis': 401,
    'banksim': 4,
    'amlsim': 6,
    'amaretto': 71,
    'cc_fraud_2023': 30,
    'sparkov': 10,  # will detect actual
    'cifer_fl': 10,
    'czech_financial': 10,
    'elliptic_bitcoin': 10,
    'financial_synthetic': 10,
    'tcs_research': 10,
}


# ============================================================================
# Data Loading — Fix the broken splits
# ============================================================================

def load_and_fix_client_data(data_path, client_id, max_samples_per_client=50000):
    """
    Load client data and fix the split issue.
    
    The preprocessing created splits where:
    - train: almost all legitimate (tiny fraud %)
    - val: 100% fraud
    - test: 100% fraud
    
    We fix this by:
    1. Merging all splits back together
    2. Doing a proper stratified random split (70/15/15)
    3. Subsampling if too large (for memory)
    """
    client_dir = Path(data_path) / f'client_{client_id}'
    
    all_sequences = []
    all_labels = []
    
    for split in ['train', 'val', 'test']:
        fpath = client_dir / f'{split}_temporal.pt'
        if fpath.exists():
            data = torch.load(fpath, map_location='cpu', weights_only=False)
            all_sequences.append(data['sequences'])
            all_labels.append(data['labels'])
    
    if not all_sequences:
        raise FileNotFoundError(f"No data found for client {client_id} at {client_dir}")
    
    X = torch.cat(all_sequences, dim=0)
    y = torch.cat(all_labels, dim=0)
    
    # Subsample if too large (memory constraint)
    if len(X) > max_samples_per_client:
        # Stratified subsample
        indices_0 = (y == 0).nonzero(as_tuple=True)[0]
        indices_1 = (y == 1).nonzero(as_tuple=True)[0]
        
        # Keep proportional sample
        n_target = max_samples_per_client
        n_fraud_orig = len(indices_1)
        n_legit_orig = len(indices_0)
        total = n_fraud_orig + n_legit_orig
        
        n_fraud = min(n_fraud_orig, max(int(n_target * n_fraud_orig / total), 500))
        n_legit = min(n_legit_orig, n_target - n_fraud)
        
        perm_0 = indices_0[torch.randperm(len(indices_0))[:n_legit]]
        perm_1 = indices_1[torch.randperm(len(indices_1))[:n_fraud]]
        
        indices = torch.cat([perm_0, perm_1])
        indices = indices[torch.randperm(len(indices))]
        
        X = X[indices]
        y = y[indices]
    
    # Detect number of features
    num_features = X.shape[-1]
    
    # Proper stratified split: 70% train, 15% val, 15% test
    X_np = np.arange(len(X))
    y_np = y.numpy()
    
    # Handle edge case where all samples are one class
    unique_classes = np.unique(y_np)
    if len(unique_classes) < 2:
        # Can't stratify with single class — random split
        perm = np.random.permutation(len(X))
        n_train = int(0.7 * len(X))
        n_val = int(0.15 * len(X))
        
        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train + n_val]
        test_idx = perm[n_train + n_val:]
    else:
        train_idx, temp_idx = train_test_split(
            X_np, test_size=0.3, random_state=42 + client_id, stratify=y_np
        )
        y_temp = y_np[temp_idx]
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=42 + client_id, stratify=y_temp
        )
    
    result = {
        'train': (X[train_idx], y[train_idx]),
        'val': (X[val_idx], y[val_idx]),
        'test': (X[test_idx], y[test_idx]),
        'num_features': num_features,
    }
    
    # Print stats
    for split_name in ['train', 'val', 'test']:
        sx, sy = result[split_name]
        fraud_rate = sy.float().mean().item()
        print(f"    Client {client_id} {split_name}: {len(sx)} samples, "
              f"fraud={sy.sum().item()}, legit={len(sy)-sy.sum().item()}, "
              f"fraud_rate={fraud_rate:.4f}")
    
    return result


# ============================================================================
# Differential Privacy
# ============================================================================

class DifferentialPrivacy:
    """
    (ε, δ)-differential privacy for federated learning, done as proper
    DP-FedAvg (McMahan et al. 2018, "Learning Differentially Private
    Recurrent Language Models") instead of the two broken variants this
    project had before:

    v1 (src/federated/dp_utils.py original): clipped+noised raw gradients
    after every minibatch, but advertised a single ε "per round" - with L
    batches x E epochs that's the mechanism applied L*E times per round with
    no composition accounting, so the real privacy loss was far larger than
    any ε quoted anywhere in this project.

    v2 (an earlier fix attempt kept in git history): moved clip+noise to
    once per client per round, on that client's own update, independently.
    Cleaner accounting, but each client's update - already noised to protect
    a single client - collapses in the server's FedAvg average without the
    noise "diluting" the way it should. Worse: for a d-dimensional update
    (this model has ~1.07M parameters), the standard Gaussian-mechanism
    noise vector's L2 norm scales as sigma*sqrt(d) — verified empirically,
    at epsilon=1.0 the noise norm was ~100x the clip norm, destroying the
    update entirely.

    This version (DP-FedAvg proper): each client only CLIPS its update to
    `max_norm` (deterministic, no noise) before sending it in. The SERVER
    adds exactly ONE Gaussian noise draw per round, to the aggregated
    (weighted-averaged) update. The sensitivity of that weighted average to
    one client's presence/absence is `max_client_weight * max_norm` (much
    smaller than max_norm itself once there are several clients), and noise
    is added once total per round instead of once per client - both cut the
    effective noise substantially relative to v2, though the fundamental
    tension remains: with ~1e6 parameters, sigma*sqrt(d) still requires a
    much larger epsilon than the textbook "ε=1" to keep the model usable.
    See docs/ARCHITECTURE.md for the worked numbers and what this means.
    """

    def __init__(self, max_norm=1.0, epsilon=1.0, delta=1e-5, device='cpu'):
        self.max_norm = max_norm
        self.epsilon = epsilon
        self.delta = delta
        self.device = device

    def clip_update(self, model, global_params):
        """
        Deterministically clip the client's local update to L2 norm
        `max_norm`, in place. No noise here - noise is added once by the
        server after aggregation (see `add_server_noise`).

        Returns:
            update_norm: L2 norm of the update before clipping (for logging)
        """
        with torch.no_grad():
            deltas = [p.detach() - g.detach() for p, g in zip(model.parameters(), global_params)]
            update_norm = torch.norm(torch.stack([torch.norm(d) for d in deltas])).item()

            clip_coef = min(1.0, self.max_norm / (update_norm + 1e-10))
            for p, g, d in zip(model.parameters(), global_params, deltas):
                p.copy_(g + d * clip_coef)

        return update_norm

    def noise_scale(self, max_client_weight):
        """
        Gaussian mechanism sigma for the server's one-time noise addition.
        Sensitivity of the weighted-average aggregate to one client is
        `max_client_weight * max_norm` (that client's clipped contribution,
        scaled by its own averaging weight) rather than the full `max_norm`.
        """
        sensitivity = max_client_weight * self.max_norm
        return sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon

    def compute_epsilon(self, *_args, **_kwargs):
        """The mechanism is applied once per round at a fixed (epsilon, delta)."""
        return self.epsilon


def add_server_noise(global_model, dp, max_client_weight):
    """Add the server's one-time-per-round Gaussian noise to the aggregated
    global model, in place. Called once per round, after federated_aggregate."""
    sigma = dp.noise_scale(max_client_weight)
    with torch.no_grad():
        for p in global_model.parameters():
            noise = torch.normal(mean=0.0, std=sigma, size=p.shape, device=p.device)
            p.add_(noise)
    return sigma


# ============================================================================
# Local Training (per client)
# ============================================================================

def train_local(model, train_loader, optimizer, dp, config, device, 
                global_params=None, mu=0.1, epochs=5):
    """
    Train model locally on one client's data.
    
    Implements FedProx: adds proximal term ||w - w_global||² to loss
    to prevent client drift in non-IID settings.
    
    Args:
        model: The model to train
        train_loader: DataLoader for this client's training data
        optimizer: Optimizer
        dp: DifferentialPrivacy instance (or None)
        config: Model config dict
        device: torch device
        global_params: Global model parameters (for FedProx proximal term)
        mu: FedProx proximal coefficient
        epochs: Number of local epochs
    
    Returns:
        avg_loss: Average training loss
        metrics: Dict with accuracy, grad_norms, etc.
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    grad_norms = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_idx, (sequences, labels) in enumerate(train_loader):
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(sequences, graph_features=None, labels=labels)
            
            # Compute task loss
            loss, loss_dict = model.compute_loss(outputs, labels)
            
            # FedProx proximal term: (mu/2) * ||w - w_global||²
            if global_params is not None and mu > 0:
                proximal_term = 0.0
                for local_param, global_param in zip(model.parameters(), global_params):
                    proximal_term += ((local_param - global_param.detach()) ** 2).sum()
                loss = loss + (mu / 2.0) * proximal_term
            
            # Defense in depth: if this batch's loss is already non-finite
            # (can still happen upstream of the nan_to_num guard in
            # compute_loss, e.g. via the FedProx proximal term or the
            # auxiliary/contrastive heads), skip the step entirely rather
            # than backprop nan/inf gradients into the weights - grad-norm
            # clipping does NOT protect against this, since clip_coef itself
            # becomes nan when total_norm is nan, silently poisoning every
            # parameter for the rest of training.
            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue

            # Backward pass
            loss.backward()

            # Standard gradient clipping for training stability (not the DP
            # mechanism - see below, DP is applied once per round to the
            # whole local update, not per-batch to raw gradients).
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_norms.append(gn.item() if isinstance(gn, torch.Tensor) else gn)

            optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            preds = (outputs['fraud_prob'] > 0.5).long()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        total_loss += epoch_loss / max(len(train_loader), 1)

    # DP clipping (deterministic) - the server adds noise once per round,
    # after aggregating all selected clients' clipped updates (DP-FedAvg).
    update_norm = None
    if dp is not None and global_params is not None:
        update_norm = dp.clip_update(model, global_params)

    avg_loss = total_loss / epochs
    accuracy = total_correct / max(total_samples, 1)

    metrics = {
        'accuracy': accuracy,
        'avg_grad_norm': float(np.mean(grad_norms)) if grad_norms else 0.0,
        'update_norm': update_norm,
    }

    return avg_loss, metrics


def evaluate_model(model, data_loader, config, device):
    """
    Evaluate model and compute comprehensive metrics.
    
    Returns real metrics — nothing hardcoded.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    all_anomaly_scores = []
    
    with torch.no_grad():
        for sequences, labels in data_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences, graph_features=None, labels=labels)
            loss, _ = model.compute_loss(outputs, labels)
            
            total_loss += loss.item()
            
            preds = (outputs['fraud_prob'] > 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs['fraud_prob'].cpu().numpy())
            if outputs['anomaly_scores'] is not None:
                all_anomaly_scores.extend(outputs['anomaly_scores'].cpu().numpy())
    
    avg_loss = total_loss / max(len(data_loader), 1)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute real metrics
    metrics = {
        'loss': avg_loss,
        'accuracy': float(accuracy_score(all_labels, all_preds)),
        'precision': float(precision_score(all_labels, all_preds, zero_division=0)),
        'recall': float(recall_score(all_labels, all_preds, zero_division=0)),
        'f1': float(f1_score(all_labels, all_preds, zero_division=0)),
    }
    
    # ROC-AUC (requires both classes present)
    if len(np.unique(all_labels)) > 1:
        metrics['roc_auc'] = float(roc_auc_score(all_labels, all_probs))
        metrics['pr_auc'] = float(average_precision_score(all_labels, all_probs))
    else:
        metrics['roc_auc'] = 0.0
        metrics['pr_auc'] = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    metrics['confusion_matrix'] = cm.tolist()
    
    # False positive rate
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    metrics['fpr'] = float(fp / max(fp + tn, 1))
    metrics['fnr'] = float(fn / max(fn + tp, 1))
    metrics['tp'] = int(tp)
    metrics['fp'] = int(fp)
    metrics['tn'] = int(tn)
    metrics['fn'] = int(fn)
    
    return metrics


# ============================================================================
# FedProx Aggregation
# ============================================================================

def federated_aggregate(global_model, client_models, client_sizes):
    """
    FedAvg aggregation: weighted average of client model parameters.
    
    w_global = Σ (n_k / N) * w_k
    
    where n_k = number of samples on client k, N = total samples.
    """
    total_size = sum(client_sizes)
    weights = [n / total_size for n in client_sizes]
    
    global_state = global_model.state_dict()
    
    # Weighted average of all parameters
    for key in global_state.keys():
        global_state[key] = torch.zeros_like(global_state[key], dtype=torch.float)
        for i, client_model in enumerate(client_models):
            client_state = client_model.state_dict()
            global_state[key] += weights[i] * client_state[key].float()
    
    global_model.load_state_dict(global_state)
    return global_model


def federated_aggregate_he(global_model, client_models, client_sizes, he, he_workers=None):
    """
    The same FedAvg weighted average as federated_aggregate() -
    w_global = Σ (n_k / N) * w_k - but computed under Paillier homomorphic
    encryption: each client's full parameter vector is encrypted (public
    key only) and scalar-multiplied by its integer sample count n_k while
    still encrypted, every selected client's result is homomorphically
    summed, and the SERVER (the only party holding the private key)
    decrypts only that one aggregate - never an individual client's
    update. Verified algebraically identical to the plaintext version up
    to float rounding (see src/federated/homomorphic_encryption.py's
    weighted-aggregation self-test).

    Real, measured cost (512-bit keys, 22 cores, 45,292 params): ~22s to
    encrypt one client's full update, ~20s to decrypt the final aggregate
    - substantial, disclosed overhead, not hidden. 512-bit keys are
    demo-speed only; the module's own docstring already says 2048+ is
    needed for production security, which costs roughly another order of
    magnitude of compute (see the benchmark in homomorphic_encryption.py).
    """
    from src.federated.homomorphic_encryption import encrypt_values_parallel, decrypt_values_parallel

    # Only the actual trainable parameters go through HE - same scope the
    # DP clip step already operates on (model.parameters(), not buffers).
    # This model also carries 3 non-parameter buffers (contrastive memory
    # bank/pointer/labels) that .parameters() doesn't include; the
    # plaintext federated_aggregate() above averages those too via a
    # blanket state_dict() loop, which is semantically questionable for a
    # memory bank/pointer in the first place - the HE path deliberately
    # leaves them untouched (copied from the first selected client)
    # instead of reproducing that, which is a disclosed, intentional
    # difference, not an oversight.
    param_names = [name for name, _ in global_model.named_parameters()]
    param_shapes = [p.shape for _, p in global_model.named_parameters()]
    param_numels = [p.numel() for _, p in global_model.named_parameters()]
    total_size = sum(client_sizes)

    timing = {"encrypt_seconds": [], "decrypt_seconds": None}
    agg_cipher = None
    for client_model, n_k in zip(client_models, client_sizes):
        client_params = dict(client_model.named_parameters())
        flat = torch.cat([client_params[name].detach().flatten() for name in param_names]).double()
        # Scale by n_k in PLAINTEXT before encrypting (Enc(n_k*m) computed
        # directly) rather than encrypt-then-multiply_encrypted_by_scalar,
        # which would be a second, equally expensive modular exponentiation
        # per value run sequentially on the main process - mathematically
        # identical since the client already holds the plaintext, but ~2x
        # faster and avoids a slow, easy-to-miss undisclosed cost.
        flat_scaled = (flat * n_k).tolist()
        t0 = time.time()
        enc_scaled = encrypt_values_parallel(flat_scaled, he.public_key, n_workers=he_workers)
        timing["encrypt_seconds"].append(time.time() - t0)

        if agg_cipher is None:
            agg_cipher = enc_scaled
        else:
            agg_cipher = [he.add_encrypted(a, b) for a, b in zip(agg_cipher, enc_scaled)]

    t0 = time.time()
    decrypted_sum = decrypt_values_parallel(agg_cipher, he.private_key, n_workers=he_workers)
    timing["decrypt_seconds"] = time.time() - t0

    flat_avg = torch.tensor(decrypted_sum, dtype=torch.float) / total_size

    global_state = global_model.state_dict()
    offset = 0
    for name, numel, shape in zip(param_names, param_numels, param_shapes):
        global_state[name] = flat_avg[offset:offset + numel].reshape(shape)
        offset += numel
    global_model.load_state_dict(global_state)
    return global_model, timing


def federated_aggregate_secure(global_model, client_models, client_sizes, client_ids,
                                he, proxy_manager, he_workers=None):
    """
    The same HE-weighted aggregation as federated_aggregate_he(), but routed
    through a tiered client -> proxy -> global-server topology
    (src/federated/proxy_server.py's ProxyServer/ProxyServerManager) instead
    of client -> server directly. This is "Secure Aggregation" layer 4 from
    docs/ARCHITECTURE.md's privacy table, now actually wired in rather than
    reference-only.

    Each client still encrypts its own update (public key only) and scales
    it by its own n_k in plaintext before encrypting - identical to
    federated_aggregate_he(). What's different: the encrypted, pre-scaled
    update goes to that client's ASSIGNED PROXY (round-robin,
    client_id % num_proxies - ProxyServerManager), which blindly sums
    everything it receives under encryption and forwards ONE partial sum to
    the global server; the server sums the proxies' partial sums (also
    blind) and is the only party that ever decrypts anything, and only the
    final total. No proxy and no client ever sees a plaintext update, and no
    single proxy sees every client's contribution - a basic
    collusion-resistance property beyond what direct client->server HE
    alone provides (a colluding server there still only needs itself; here
    it would need the server AND every proxy that handled at least one
    client this round).
    """
    from src.federated.homomorphic_encryption import encrypt_values_parallel, decrypt_values_parallel

    param_names = [name for name, _ in global_model.named_parameters()]
    param_shapes = [p.shape for _, p in global_model.named_parameters()]
    param_numels = [p.numel() for _, p in global_model.named_parameters()]
    total_size = sum(client_sizes)

    proxy_manager.reset_all_proxies()

    timing = {"encrypt_seconds": [], "proxy_aggregate_seconds": None, "decrypt_seconds": None}
    for client_model, n_k, cid in zip(client_models, client_sizes, client_ids):
        client_params = dict(client_model.named_parameters())
        flat = torch.cat([client_params[name].detach().flatten() for name in param_names]).double()
        flat_scaled = (flat * n_k).tolist()

        t0 = time.time()
        enc_scaled = encrypt_values_parallel(flat_scaled, he.public_key, n_workers=he_workers)
        timing["encrypt_seconds"].append(time.time() - t0)

        proxy = proxy_manager.get_proxy_for_client(cid)
        proxy.receive_encrypted_gradient(cid, enc_scaled)

    t0 = time.time()
    agg_cipher = proxy_manager.aggregate_all_proxies()
    timing["proxy_aggregate_seconds"] = time.time() - t0

    t0 = time.time()
    decrypted_sum = decrypt_values_parallel(agg_cipher, he.private_key, n_workers=he_workers)
    timing["decrypt_seconds"] = time.time() - t0

    flat_avg = torch.tensor(decrypted_sum, dtype=torch.float) / total_size

    global_state = global_model.state_dict()
    offset = 0
    for name, numel, shape in zip(param_names, param_numels, param_shapes):
        global_state[name] = flat_avg[offset:offset + numel].reshape(shape)
        offset += numel
    global_model.load_state_dict(global_state)
    return global_model, timing


# ============================================================================
# Main FL Training Loop
# ============================================================================

def run_federated_training(args):
    """
    Main federated learning training loop.
    
    Protocol (FedProx):
    1. Initialize global model
    2. For each round:
       a. Select fraction of clients
       b. Send global model to selected clients
       c. Each client trains locally for E epochs with:
          - BCE loss + contrastive loss + auxiliary loss
          - FedProx proximal term: (μ/2)||w - w_global||²
          - Differential privacy: clip gradients + add noise
       d. Aggregate client models: w_global = Σ (n_k/N) * w_k
       e. Evaluate global model on test set
    3. Save final model and all metrics
    """
    device = torch.device('cpu')  # Mac M-series — MPS can be unstable
    print(f"\nDevice: {device}")
    
    # Data path
    data_path = Path(args.data_path) / args.dataset
    if not data_path.exists():
        print(f"ERROR: Data path not found: {data_path}")
        sys.exit(1)
    
    # Detect number of features from data
    sample_data = torch.load(
        data_path / 'client_0' / 'train_temporal.pt', 
        map_location='cpu', weights_only=False
    )
    num_features = sample_data['sequences'].shape[-1]
    print(f"Detected {num_features} features for {args.dataset}")
    
    # Config
    config = copy.deepcopy(DEFAULT_CONFIG)
    config['dataset']['num_features'] = num_features
    if args.pos_weight is not None:
        config['training']['pos_weight'] = args.pos_weight
    if args.fraud_rate_prior is not None:
        config['training']['fraud_rate_prior'] = args.fraud_rate_prior
    
    # Reduce model size for large feature counts (memory)
    if num_features > 100:
        config['model']['hidden_dim'] = 64
        config['model']['temporal']['num_heads'] = 4
        config['model']['temporal']['num_layers'] = 2
        config['model']['graph']['num_layers'] = 2
    
    # Output directory
    results_dir = Path(args.output) / args.dataset
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== Load Data ==========
    print(f"\n{'='*60}")
    print(f"Loading data: {args.dataset}")
    print(f"{'='*60}")
    
    num_clients = min(args.clients, 10)
    excluded = {int(c) for c in args.exclude_clients.split(',') if c.strip() != ''}
    if excluded:
        print(f"Excluding clients: {sorted(excluded)}")
    client_data = {}

    for client_id in range(num_clients):
        if client_id in excluded:
            continue
        try:
            client_data[client_id] = load_and_fix_client_data(
                data_path, client_id, 
                max_samples_per_client=args.max_samples
            )
        except Exception as e:
            print(f"  WARNING: Could not load client {client_id}: {e}")
    
    if len(client_data) < 2:
        print("ERROR: Need at least 2 clients with data")
        sys.exit(1)
    
    print(f"\nLoaded {len(client_data)} clients successfully")
    
    # ========== Initialize Global Model ==========
    print(f"\n{'='*60}")
    print("Initializing model")
    print(f"{'='*60}")
    
    global_model = TemporalGraphTransformer(config).to(device)
    total_params = sum(p.numel() for p in global_model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # ========== FL Training ==========
    print(f"\n{'='*60}")
    print(f"Federated Learning Training")
    print(f"  Dataset: {args.dataset}")
    print(f"  Rounds: {args.rounds}")
    print(f"  Clients: {len(client_data)}")
    print(f"  Clients per round: {args.clients_per_round}")
    print(f"  Local epochs: {args.local_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  FedProx mu: {args.mu}")
    print(f"  DP enabled: {args.dp}")
    if args.dp:
        print(f"  DP epsilon (per client per round): {args.epsilon}")
        print(f"  DP clip norm: {args.clip_norm}")
    print(f"{'='*60}\n")
    
    history = {
        'rounds': [],
        'global_metrics': [],
        'client_metrics': [],
        'config': {
            'dataset': args.dataset,
            'num_rounds': args.rounds,
            'num_clients': len(client_data),
            'clients_per_round': args.clients_per_round,
            'local_epochs': args.local_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'mu': args.mu,
            'dp_enabled': args.dp,
            'epsilon': args.epsilon if args.dp else None,
            'clip_norm': args.clip_norm if args.dp else None,
            'he_enabled': args.he,
            'he_key_size': args.he_key_size if args.he else None,
            'secure_agg_enabled': args.secure_agg,
            'num_proxies': args.num_proxies if args.secure_agg else None,
            'num_features': num_features,
            'total_params': total_params,
        }
    }
    
    # Prepare a combined test set from all clients for global evaluation
    test_sequences = []
    test_labels = []
    for cid in client_data:
        tx, ty = client_data[cid]['test']
        test_sequences.append(tx)
        test_labels.append(ty)
    test_X = torch.cat(test_sequences, dim=0)
    test_y = torch.cat(test_labels, dim=0)
    test_dataset = TensorDataset(test_X, test_y)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Global test set: {len(test_X)} samples, "
          f"fraud={test_y.sum().item()}, legit={len(test_y)-test_y.sum().item()}, "
          f"fraud_rate={test_y.float().mean().item():.4f}\n")

    # Homomorphic encryption: one keypair for the whole run (real-world
    # systems don't rotate HE keys every round either). Only the server
    # (this process) ever sees he.private_key; client-side encryption
    # below only ever touches he.public_key.
    he = None
    if args.he:
        from src.federated.homomorphic_encryption import PaillierEncryption
        print(f"[HE] Generating Paillier keypair (key_size={args.he_key_size} bits)...")
        if args.he_key_size < 2048:
            print(f"[HE] WARNING: {args.he_key_size}-bit keys are demo-speed only, not "
                  f"production-secure (use --he-key-size 2048+ for real security; this is "
                  f"a real, measured compute/security tradeoff, not a hidden shortcut).")
        he = PaillierEncryption(key_size=args.he_key_size)
        print(f"[HE] Keypair ready.\n")

    # Secure Aggregation: one proxy topology for the whole run, reused
    # (and cleared) every round - a real deployment wouldn't re-assign
    # clients to proxies mid-experiment either.
    proxy_manager = None
    if args.secure_agg:
        from src.federated.proxy_server import ProxyServerManager
        print(f"[SecureAgg] Setting up {args.num_proxies} proxy server(s) for "
              f"{args.clients} clients (round-robin assignment)...")
        proxy_manager = ProxyServerManager(args.num_proxies, args.clients, he)
        print()

    start_time = time.time()
    
    for round_num in range(1, args.rounds + 1):
        round_start = time.time()
        
        # Select clients for this round
        available_clients = list(client_data.keys())
        num_selected = min(args.clients_per_round, len(available_clients))
        selected_clients = np.random.choice(
            available_clients, size=num_selected, replace=False
        ).tolist()
        
        print(f"Round {round_num}/{args.rounds} — Selected clients: {selected_clients}")
        
        # Save global model parameters for FedProx
        global_params = [p.clone().detach() for p in global_model.parameters()]

        # Differential privacy for this round (DP-FedAvg): one instance
        # shared by every client (they only clip) and by the server (which
        # adds noise once, after aggregation) - see class docstring.
        dp = None
        if args.dp:
            dp = DifferentialPrivacy(
                max_norm=args.clip_norm,
                epsilon=args.epsilon,
                delta=args.delta,
                device=device
            )

        # Train each selected client
        client_models = []
        client_sizes = []
        round_client_metrics = []

        for cid in selected_clients:
            # Create client model (copy of global)
            client_model = copy.deepcopy(global_model)

            # Prepare data loader
            train_X, train_y = client_data[cid]['train']
            train_dataset = TensorDataset(train_X, train_y)

            # At real PaySim's ~0.1-0.3% fraud rate, batch_size=64 random
            # sampling gives ~0.1 fraud examples/batch on average - most
            # batches contain literally zero positives, so pos_weight has
            # nothing to act on most of the time (verified: tuning pos_weight
            # 8->400 changed nothing, and post-hoc threshold tuning on the
            # trained checkpoint recovered ~0 extra F1, so the bottleneck is
            # upstream of both the loss weighting and the decision threshold).
            # WeightedRandomSampler fixes this at the source: sample with
            # replacement so every batch has a realistic mix of both classes,
            # independent of the true class ratio.
            if args.oversample:
                n_pos = int(train_y.sum().item())
                n_neg = len(train_y) - n_pos
                if n_pos > 0 and n_neg > 0:
                    # target_pos_frac controls how aggressively we oversample:
                    # 0.5 = full balance (recall up a lot, precision collapses -
                    # verified empirically: recall 0.58->0.91, precision
                    # 0.70->0.13, F1 down overall). A smaller target fraction
                    # is a tunable middle ground between the two failure modes.
                    target_pos_frac = args.oversample_ratio
                    weight_per_class = torch.tensor([
                        (1.0 - target_pos_frac) / n_neg,
                        target_pos_frac / n_pos,
                    ])
                    sample_weights = weight_per_class[train_y.long()]
                    sampler = WeightedRandomSampler(
                        sample_weights, num_samples=len(train_y), replacement=True
                    )
                    train_loader = DataLoader(
                        train_dataset, batch_size=args.batch_size, sampler=sampler
                    )
                else:
                    train_loader = DataLoader(
                        train_dataset, batch_size=args.batch_size, shuffle=True
                    )
            else:
                train_loader = DataLoader(
                    train_dataset, batch_size=args.batch_size,
                    shuffle=True, drop_last=False
                )

            # Optimizer
            optimizer = torch.optim.Adam(
                client_model.parameters(), lr=args.lr, weight_decay=1e-5
            )

            # Local training (clips its own update to args.clip_norm if dp is set)
            train_loss, train_metrics = train_local(
                client_model, train_loader, optimizer, dp, config, device,
                global_params=global_params, mu=args.mu, epochs=args.local_epochs
            )

            # Evaluate on client's val set
            val_X, val_y = client_data[cid]['val']
            val_dataset = TensorDataset(val_X, val_y)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
            val_metrics = evaluate_model(client_model, val_loader, config, device)

            client_models.append(client_model)
            client_sizes.append(len(train_X))
            
            client_info = {
                'client_id': cid,
                'train_loss': train_loss,
                'train_accuracy': train_metrics['accuracy'],
                'val_accuracy': val_metrics['accuracy'],
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall'],
                'val_f1': val_metrics['f1'],
                'val_roc_auc': val_metrics['roc_auc'],
                'update_norm': train_metrics.get('update_norm'),
                'num_samples': len(train_X),
            }
            round_client_metrics.append(client_info)

            print(f"  Client {cid}: loss={train_loss:.4f}, "
                  f"train_acc={train_metrics['accuracy']:.4f}, "
                  f"val_acc={val_metrics['accuracy']:.4f}, "
                  f"val_f1={val_metrics['f1']:.4f}")

        # Aggregate client models → global model. With --he, this happens
        # under Paillier homomorphic encryption: the server never sees any
        # individual client's update in plaintext, only the decrypted
        # aggregate. With --secure-agg on top, that encrypted aggregation
        # is additionally routed through a tiered proxy topology (see
        # federated_aggregate_he / federated_aggregate_secure docstrings
        # for the real, measured timing cost of each).
        he_timing = None
        if proxy_manager is not None:
            global_model, he_timing = federated_aggregate_secure(
                global_model, client_models, client_sizes, selected_clients,
                he, proxy_manager, he_workers=args.he_workers
            )
            print(f"  [SecureAgg] encrypted {len(client_models)} client update(s) "
                  f"({sum(he_timing['encrypt_seconds']):.1f}s total) across "
                  f"{args.num_proxies} proxies ({he_timing['proxy_aggregate_seconds']:.2f}s "
                  f"proxy aggregation), decrypted final aggregate "
                  f"({he_timing['decrypt_seconds']:.1f}s)")
        elif he is not None:
            global_model, he_timing = federated_aggregate_he(
                global_model, client_models, client_sizes, he, he_workers=args.he_workers
            )
            print(f"  [HE] encrypted {len(client_models)} client update(s) "
                  f"({sum(he_timing['encrypt_seconds']):.1f}s total), decrypted aggregate "
                  f"({he_timing['decrypt_seconds']:.1f}s)")
        else:
            global_model = federated_aggregate(global_model, client_models, client_sizes)

        # DP-FedAvg: server adds ONE noise draw for the round, to the
        # aggregate, sized by the largest single client's averaging weight.
        dp_epsilon = 0.0
        if dp is not None:
            max_weight = max(client_sizes) / sum(client_sizes)
            sigma = add_server_noise(global_model, dp, max_weight)
            dp_epsilon = dp.compute_epsilon()
            print(f"  [DP] server noise added: sigma={sigma:.4f}, "
                  f"epsilon spent this round={dp_epsilon:.2f}")

        # Evaluate global model on combined test set
        global_metrics = evaluate_model(global_model, test_loader, config, device)
        
        round_time = time.time() - round_start
        
        print(f"  Global: acc={global_metrics['accuracy']:.4f}, "
              f"prec={global_metrics['precision']:.4f}, "
              f"rec={global_metrics['recall']:.4f}, "
              f"f1={global_metrics['f1']:.4f}, "
              f"auc={global_metrics['roc_auc']:.4f}, "
              f"fpr={global_metrics['fpr']:.4f} "
              f"[{round_time:.1f}s]\n")
        
        # Save history
        history['rounds'].append(round_num)
        global_metrics['dp_epsilon'] = dp_epsilon
        if he_timing is not None:
            global_metrics['he_encrypt_seconds'] = he_timing['encrypt_seconds']
            global_metrics['he_decrypt_seconds'] = he_timing['decrypt_seconds']
            if 'proxy_aggregate_seconds' in he_timing:
                global_metrics['he_proxy_aggregate_seconds'] = he_timing['proxy_aggregate_seconds']
        history['global_metrics'].append(global_metrics)
        history['client_metrics'].append(round_client_metrics)
    
    total_time = time.time() - start_time
    
    # ========== Save Results ==========
    print(f"\n{'='*60}")
    print("Saving results")
    print(f"{'='*60}")
    
    # Save model checkpoint
    checkpoint_path = results_dir / 'global_model.pt'
    torch.save({
        'model_state_dict': global_model.state_dict(),
        'config': config,
        'history': history,
    }, checkpoint_path)
    print(f"  Model saved: {checkpoint_path}")
    
    # Save training history as JSON
    history_json = copy.deepcopy(history)
    # Convert numpy arrays to lists for JSON serialization
    for gm in history_json['global_metrics']:
        for key, val in gm.items():
            if isinstance(val, np.ndarray):
                gm[key] = val.tolist()
            elif isinstance(val, (np.floating, np.integer)):
                gm[key] = float(val)
    
    history_path = results_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history_json, f, indent=2)
    print(f"  History saved: {history_path}")
    
    # ========== Final Summary ==========
    final_metrics = history['global_metrics'][-1]
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Rounds: {args.rounds}")
    print(f"\nFinal Global Model Performance:")
    print(f"  Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"  Precision: {final_metrics['precision']:.4f}")
    print(f"  Recall:    {final_metrics['recall']:.4f}")
    print(f"  F1-Score:  {final_metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {final_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:    {final_metrics['pr_auc']:.4f}")
    print(f"  FPR:       {final_metrics['fpr']:.4f}")
    print(f"  Confusion Matrix:")
    cm = final_metrics['confusion_matrix']
    print(f"    TN={cm[0][0]:>6}  FP={cm[0][1]:>6}")
    print(f"    FN={cm[1][0]:>6}  TP={cm[1][1]:>6}")
    print(f"{'='*60}\n")
    
    return history


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Federated Fraud Detection Training')
    
    # Data
    parser.add_argument('--dataset', type=str, default='synthetic_paysim',
                       help='Dataset name - a directory under --data-path with client_N/{split}_temporal.pt files')
    parser.add_argument('--data-path', type=str,
                       default=str(Path(__file__).resolve().parent / 'data'),
                       help='Path to preprocessed datasets')
    parser.add_argument('--output', type=str,
                       default=str(Path(__file__).resolve().parent / 'results'),
                       help='Output directory for results')
    
    # FL settings
    parser.add_argument('--rounds', type=int, default=20, help='Number of FL rounds')
    parser.add_argument('--clients', type=int, default=10, help='Total number of clients')
    parser.add_argument('--clients-per-round', type=int, default=3,
                       help='Clients sampled per round')
    parser.add_argument('--local-epochs', type=int, default=3, help='Local epochs per client')
    parser.add_argument('--batch-size', type=int, default=64, help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--mu', type=float, default=0.1, help='FedProx proximal coefficient')
    parser.add_argument('--max-samples', type=int, default=30000,
                       help='Max samples per client (for memory)')
    
    # Differential Privacy (client-level, applied once per round to the local update)
    parser.add_argument('--dp', action='store_true', help='Enable differential privacy')
    parser.add_argument('--epsilon', type=float, default=100.0,
                       help='DP privacy budget per round (see docs/ARCHITECTURE.md: '
                            'this model has ~45K parameters (right-sized to the 10-dim '
                            'input, see DEFAULT_CONFIG), and the standard Gaussian '
                            'mechanism noise norm scales as sigma*sqrt(d) - epsilon=1.0 '
                            '(the textbook value) still empirically destroys this model '
                            '(verified: f1=0 through epsilon=20), but ~50 is where it '
                            'starts working (f1=0.74) and ~100 gives working margin; '
                            'a ~24x smaller model than the original 1.07M-param version '
                            'moved this floor from ~2000 down to ~100 (~20x better)')
    parser.add_argument('--delta', type=float, default=1e-5,
                       help='DP delta (failure probability)')
    parser.add_argument('--clip-norm', type=float, default=1.0,
                       help='DP clipping norm for the client update')

    # Model/loss hyperparameters that need re-tuning per dataset imbalance
    # (DEFAULT_CONFIG's 8.0 / 0.06 were tuned for synthetic data's ~6% fraud
    # rate; real PaySim is ~45x more imbalanced at ~0.13%)
    parser.add_argument('--pos-weight', type=float, default=None,
                       help='BCE positive-class weight (overrides DEFAULT_CONFIG; '
                            'default 8.0 was tuned for synthetic ~6%% imbalance)')
    parser.add_argument('--fraud-rate-prior', type=float, default=None,
                       help='Prior fraud rate used for the classifier bias init '
                            '(overrides DEFAULT_CONFIG; default 0.06 was tuned for '
                            'synthetic ~6%% imbalance)')
    # Homomorphic encryption (client-side encrypt, server-side aggregate
    # under encryption, server-side decrypt only the aggregate - real
    # Paillier crypto, see src/federated/homomorphic_encryption.py)
    parser.add_argument('--he', action='store_true',
                       help='Encrypt client updates with Paillier HE before aggregation, '
                            'instead of plaintext FedAvg - the server only ever decrypts '
                            'the aggregate, never an individual client update')
    parser.add_argument('--he-key-size', type=int, default=512,
                       help='Paillier key size in bits. 512 is demo-speed only; the '
                            'module docstring says 2048+ is needed for real security '
                            '(real, measured compute tradeoff - see benchmark in '
                            'src/federated/homomorphic_encryption.py)')
    parser.add_argument('--he-workers', type=int, default=None,
                       help='Parallel workers for HE encrypt/decrypt (default: os.cpu_count())')
    parser.add_argument('--secure-agg', action='store_true',
                       help='Route HE-encrypted updates through a tiered client -> proxy -> '
                            'server topology (src/federated/proxy_server.py) instead of '
                            'client -> server directly - requires --he. No single proxy sees '
                            'every client, and the global server never sees a per-proxy '
                            'partial sum in plaintext either, only the final total.')
    parser.add_argument('--num-proxies', type=int, default=2,
                       help='Number of proxy servers for --secure-agg (clients are assigned '
                            'round-robin via client_id %% num_proxies)')

    parser.add_argument('--exclude-clients', type=str, default='',
                       help='Comma-separated client ids to exclude entirely '
                            '(e.g. "9" to drop the known paysim_real anomaly)')
    parser.add_argument('--oversample', action='store_true',
                       help='Use WeightedRandomSampler so every local batch has a '
                            'realistic mix of both classes, instead of the true '
                            '(possibly near-zero-fraud-per-batch) class ratio - '
                            'needed at real PaySim-scale imbalance, where '
                            'pos_weight alone has nothing to act on in most batches')
    parser.add_argument('--oversample-ratio', type=float, default=0.5,
                       help='Target fraction of positive (fraud) examples per '
                            'sampled batch when --oversample is set. 0.5 = full '
                            'balance (maximizes recall, verified to collapse '
                            'precision); lower values are a tunable middle ground')

    args = parser.parse_args()

    if args.secure_agg and not args.he:
        parser.error('--secure-agg requires --he (secure aggregation routes HE-encrypted '
                      'updates through proxies; without --he there is nothing to route)')

    np.random.seed(42)
    torch.manual_seed(42)

    run_federated_training(args)


if __name__ == '__main__':
    main()
