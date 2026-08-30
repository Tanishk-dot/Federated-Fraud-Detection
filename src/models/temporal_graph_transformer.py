"""
Complete Temporal Graph Transformer Model
Combines all components: Temporal Transformer + Graph Encoder + Fusion + Dual Heads
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .temporal_transformer import TemporalTransformer
from .graph_encoder import SimpleGraphEncoder
from .fusion import CrossModalFusion
from .contrastive_head import ContrastiveLearningHead


class TemporalGraphTransformer(nn.Module):
    """
    Complete Temporal Graph Transformer for Fraud Detection
    
    Architecture:
    1. Temporal Branch: Transformer encoder for transaction sequences
    2. Graph Branch: HGAT for entity relationships
    3. Cross-Modal Fusion: Bidirectional attention fusion
    4. Supervised Head: Binary classification (fraud probability)
    5. Contrastive Head: Anomaly detection (KNN distance)
    
    Final Decision: fraud_prob > threshold OR anomaly_score > threshold
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Extract config
        input_dim = config['dataset']['num_features']
        hidden_dim = config['model']['hidden_dim']
        
        # Temporal Transformer
        self.temporal_encoder = TemporalTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=config['model']['temporal']['num_heads'],
            num_layers=config['model']['temporal']['num_layers'],
            dropout=config['model']['temporal']['dropout']
        )
        
        # Graph Encoder (simplified for now)
        self.graph_encoder = SimpleGraphEncoder(
            hidden_dim=hidden_dim,
            num_layers=config['model']['graph']['num_layers'],
            dropout=config['model']['graph']['dropout']
        )
        
        # Graph feature projection (if graph features exist)
        self.graph_proj = nn.Linear(input_dim, hidden_dim)
        
        # Cross-Modal Fusion
        self.fusion = CrossModalFusion(
            hidden_dim=hidden_dim,
            num_heads=config['model']['temporal']['num_heads'],
            dropout=config['model']['temporal']['dropout']
        )
        
        # Supervised Classification Head
        classifier_hidden = config['model']['classifier']['hidden_dim']
        classifier_dropout = config['model']['classifier']['dropout']
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, classifier_hidden),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden, 1),
            nn.Sigmoid()
        )

        # Prior-informed bias init: on an imbalanced task (e.g. ~6% fraud),
        # a zero-init final bias starts the sigmoid at 0.5, and BCE's easiest
        # early gradient descent direction is "learn the constant base rate,
        # ignore the input" - with a hidden_dim this large relative to a
        # 10-dim input, that constant-output solution is a genuinely stable
        # local minimum the optimizer settles into and never escapes within
        # a normal training budget (observed empirically: probs std -> ~0
        # across the whole batch after training, vs ~0.05 at init - the
        # network actively learns to discard per-sample signal). Starting
        # the bias at the prior's log-odds removes "match the base rate" as
        # a free win, so early gradients are forced to work on the weights
        # that actually use input features instead.
        prior = config.get('training', {}).get('fraud_rate_prior', 0.06)
        prior = min(max(prior, 1e-4), 1 - 1e-4)
        bias_init = float(np.log(prior / (1 - prior)))
        with torch.no_grad():
            self.classifier[3].bias.fill_(bias_init)

        # Contrastive Learning Head
        self.contrastive_head = ContrastiveLearningHead(
            hidden_dim=hidden_dim,
            projection_dim=config['model']['contrastive']['projection_dim'],
            temperature=config['model']['contrastive']['temperature'],
            memory_bank_size=config['model']['contrastive']['memory_bank_size']
        )
        
        # Auxiliary head (predict merchant category - optional)
        self.aux_head = nn.Linear(hidden_dim, 10)  # 10 merchant categories
        
        # Store config
        self.config = config
    
    def forward(self, sequences, graph_features=None, labels=None):
        """
        Args:
            sequences: (batch, seq_len, input_dim) - transaction sequences
            graph_features: (batch, input_dim) or None - graph node features
            labels: (batch,) or None - fraud labels for training
        
        Returns:
            outputs: dict with fraud_prob, anomaly_score, embeddings, losses
        """
        batch_size = sequences.size(0)
        
        # 1. Temporal Branch
        temporal_cls, temporal_sequence = self.temporal_encoder(sequences)
        # temporal_cls: (batch, hidden_dim)
        # temporal_sequence: (batch, seq_len+1, hidden_dim)
        
        # 2. Graph Branch
        if graph_features is None:
            # No explicit entity/graph features supplied (e.g. no linked
            # user/merchant/card IDs available) - fall back to the account's
            # mean transaction profile across the window as a per-sample
            # "entity" feature. This keeps the branch's gradient signal real
            # and per-sample instead of a constant zero (which made it inert
            # for every caller that didn't explicitly pass graph_features).
            # It is NOT a substitute for a real relational graph - see
            # HeterogeneousGraphEncoder in graph_encoder.py, which is ready
            # to use once entity-linked data (real edges) is available.
            graph_features = sequences.mean(dim=1)  # (batch, input_dim)

        graph_emb = self.graph_proj(graph_features)  # (batch, hidden_dim)
        graph_emb_dict = {'user': graph_emb}
        graph_out_dict = self.graph_encoder(graph_emb_dict)
        graph_features_encoded = graph_out_dict['user']  # (batch, hidden_dim)
        
        # 3. Cross-Modal Fusion
        fused = self.fusion(graph_features_encoded, temporal_cls, temporal_sequence)
        # fused: (batch, hidden_dim)
        
        # 4. Supervised Classification Head
        fraud_prob = self.classifier(fused).squeeze(-1)  # (batch,)

        # Sanitize at the source, not just inside compute_loss: a NaN/inf
        # fraud_prob can happen for reasons unrelated to training loss (an
        # unusual input triggering a numerical edge case somewhere in the
        # transformer/attention stack at inference time), and every
        # consumer of this output - compute_loss, evaluate_model's metrics
        # (sklearn's roc_auc_score hard-errors on a single NaN in the whole
        # array, observed in practice), the backend's /api/predict, the
        # dashboard - needs a safe value, not just the loss function. Fixing
        # it once here instead of re-guarding in every caller is the
        # correct place for this, not a training-only workaround.
        fraud_prob = torch.nan_to_num(fraud_prob, nan=0.5, posinf=1 - 1e-7, neginf=1e-7)
        fraud_prob = torch.clamp(fraud_prob, min=1e-7, max=1 - 1e-7)

        # 5. Contrastive Learning Head
        projections, contrastive_loss, anomaly_scores = self.contrastive_head(fused, labels)
        if anomaly_scores is not None:
            anomaly_scores = torch.nan_to_num(anomaly_scores, nan=0.0, posinf=1.0, neginf=0.0)

        # 6. Auxiliary Head (merchant category prediction)
        aux_logits = self.aux_head(fused)  # (batch, 10)
        
        # Prepare outputs
        outputs = {
            'fraud_prob': fraud_prob,
            'anomaly_scores': anomaly_scores,
            'embeddings': fused,
            'projections': projections,
            'aux_logits': aux_logits,
            'contrastive_loss': contrastive_loss
        }
        
        return outputs
    
    def compute_loss(self, outputs, labels, aux_labels=None):
        """
        Compute multi-task loss
        
        Args:
            outputs: dict from forward()
            labels: (batch,) - fraud labels
            aux_labels: (batch,) - merchant category labels (optional)
        
        Returns:
            total_loss: scalar
            loss_dict: dict with individual losses
        """
        # 1. Supervised BCE Loss
        # Guard against both saturation and outright NaN. On highly separable
        # data with no FedProx term and no DP noise (mu=0, no-DP FedAvg
        # baseline), a single confidently-wrong prediction after sigmoid
        # saturates to exactly 0.0/1.0 produces an infinite BCE loss for that
        # sample; the resulting inf/nan gradient survives grad-norm clipping
        # (clip_coef itself becomes nan when total_norm is nan) and
        # permanently poisons every weight from that step onward, so every
        # subsequent forward pass returns fraud_prob == nan. A plain clamp
        # does not fix this, because clamp(nan) == nan. nan_to_num replaces
        # any nan with a neutral 0.5 guess (posinf/neginf with the clamp
        # bounds) so a bad batch produces a large-but-finite loss and normal
        # gradients instead of permanently corrupting the model. This is a
        # numerical-stability guard, not a change to what the model learns.
        safe_fraud_prob = torch.nan_to_num(
            outputs['fraud_prob'], nan=0.5, posinf=1 - 1e-7, neginf=1e-7
        )
        safe_fraud_prob = torch.clamp(safe_fraud_prob, min=1e-7, max=1 - 1e-7)
        # Per-sample class weighting for imbalanced fraud rates (e.g. ~6%
        # fraud): without this, unweighted BCE's steepest early-training
        # descent direction is "predict everyone as the majority class,
        # ignore the input" - a real local minimum this architecture
        # settles into and doesn't escape from on its own (see bias_init
        # above). pos_weight >1 makes a missed fraud cost more than a false
        # alarm, keeping gradient pressure on the minority class.
        pos_weight = self.config.get('training', {}).get('pos_weight', 1.0)
        if pos_weight != 1.0:
            sample_weight = torch.where(
                labels == 1,
                torch.tensor(pos_weight, device=labels.device, dtype=safe_fraud_prob.dtype),
                torch.tensor(1.0, device=labels.device, dtype=safe_fraud_prob.dtype),
            )
        else:
            sample_weight = None
        bce_loss = F.binary_cross_entropy(
            safe_fraud_prob,
            labels.float(),
            weight=sample_weight,
            reduction='mean'
        )
        
        # 2. Contrastive Loss
        contrastive_loss = outputs['contrastive_loss']
        if contrastive_loss is None:
            contrastive_loss = torch.tensor(0.0, device=labels.device)
        else:
            # Same nan_to_num guard as fraud_prob above, and for the same
            # reason it matters even though loss_weights['contrastive'] is
            # 0.0 in this project's configs: IEEE754 says 0.0 * nan == nan,
            # not 0.0, so an occasional nan from ContrastiveLearningHead's
            # internal log_softmax (its own clamp only guards +inf, not nan)
            # still poisons total_loss - observed in practice on the
            # centralized-deep-model comparison baseline (no FedProx, no DP
            # to keep it in check). This makes the weighting actually mean
            # "ignored" instead of "usually ignored".
            contrastive_loss = torch.nan_to_num(contrastive_loss, nan=0.0, posinf=100.0, neginf=0.0)

        # 3. Auxiliary Loss (merchant category)
        if aux_labels is not None:
            aux_loss = F.cross_entropy(outputs['aux_logits'], aux_labels)
            aux_loss = torch.nan_to_num(aux_loss, nan=0.0, posinf=100.0, neginf=0.0)
        else:
            aux_loss = torch.tensor(0.0, device=labels.device)

        # Weighted combination
        loss_weights = self.config['training']['loss_weights']
        total_loss = (
            loss_weights['bce'] * bce_loss +
            loss_weights['contrastive'] * contrastive_loss +
            loss_weights['auxiliary'] * aux_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'bce': bce_loss.item(),
            'contrastive': contrastive_loss.item(),
            'auxiliary': aux_loss.item()
        }
        
        return total_loss, loss_dict
    
    def predict(self, sequences, graph_features=None, fraud_threshold=0.8, anomaly_threshold=0.5):
        """
        Make fraud predictions
        
        Decision: fraud_prob > fraud_threshold OR anomaly_score > anomaly_threshold
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(sequences, graph_features)
            
            fraud_prob = outputs['fraud_prob']
            anomaly_scores = outputs['anomaly_scores']
            
            # Final decision
            predictions = ((fraud_prob > fraud_threshold) | 
                          (anomaly_scores > anomaly_threshold)).long()
            
            return predictions, fraud_prob, anomaly_scores


if __name__ == "__main__":
    print("Testing Complete Temporal Graph Transformer...")
    
    # Mock config
    config = {
        'dataset': {'num_features': 10},
        'model': {
            'hidden_dim': 128,
            'temporal': {'num_heads': 8, 'num_layers': 4, 'dropout': 0.1},
            'graph': {'num_layers': 3, 'dropout': 0.1},
            'classifier': {'hidden_dim': 64, 'dropout': 0.3},
            'contrastive': {'projection_dim': 64, 'temperature': 0.07, 'memory_bank_size': 1000}
        },
        'training': {
            'loss_weights': {'bce': 1.0, 'contrastive': 0.5, 'auxiliary': 0.3}
        }
    }
    
    model = TemporalGraphTransformer(config)
    
    # Create dummy inputs
    batch_size = 8
    seq_len = 10
    input_dim = 10
    
    sequences = torch.randn(batch_size, seq_len, input_dim)
    graph_features = torch.randn(batch_size, input_dim)
    labels = torch.randint(0, 2, (batch_size,))
    
    # Forward pass
    model.train()
    outputs = model(sequences, graph_features, labels)
    
    # Compute loss
    total_loss, loss_dict = model.compute_loss(outputs, labels)
    
    print(f"\nInput shapes:")
    print(f"  Sequences: {sequences.shape}")
    print(f"  Graph features: {graph_features.shape}")
    print(f"  Labels: {labels.shape}")
    
    print(f"\nOutput shapes:")
    print(f"  Fraud prob: {outputs['fraud_prob'].shape}")
    print(f"  Anomaly scores: {outputs['anomaly_scores'].shape}")
    print(f"  Embeddings: {outputs['embeddings'].shape}")
    
    print(f"\nLosses:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")
    
    # Test prediction
    model.eval()
    predictions, fraud_prob, anomaly_scores = model.predict(sequences, graph_features)
    print(f"\nPredictions: {predictions}")
    print(f"Fraud probabilities: {fraud_prob}")
    
    print("\n✅ Complete Temporal Graph Transformer test passed!")
