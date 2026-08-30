"""
Flower Federated Learning Client
Handles local training and communication with FL server
"""

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Tuple
import flwr as fl

from ..models.temporal_graph_transformer import TemporalGraphTransformer
from ..data.dataset import FraudDetectionDataModule
from .dp_utils import DifferentialPrivacy


class FraudDetectionClient(fl.client.NumPyClient):
    """
    Flower FL Client for Fraud Detection
    
    Responsibilities:
    1. Receive global model weights from server
    2. Train locally for N epochs
    3. Apply differential privacy to gradients
    4. Send updated weights back to server
    5. Evaluate on local validation data
    """
    
    def __init__(self, client_id: int, config: dict, device: str = 'cpu'):
        """
        Args:
            client_id: Unique client identifier (0-9)
            config: Experiment configuration
            device: 'cpu' or 'cuda'
        """
        self.client_id = client_id
        self.config = config
        self.device = torch.device(device)
        
        print(f"\n{'='*60}")
        print(f"Initializing FL Client {client_id}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        # Initialize model
        self.model = TemporalGraphTransformer(config).to(self.device)
        
        # Initialize data module
        self.data_module = FraudDetectionDataModule(config, client_id)
        self.data_module.setup()
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Initialize differential privacy
        if config['privacy']['enabled']:
            self.dp = DifferentialPrivacy(
                epsilon=config['privacy']['epsilon'],
                delta=config['privacy']['delta'],
                clip_norm=config['privacy']['clip_norm']
            )
        else:
            self.dp = None
        
        # Training config
        self.local_epochs = config['training']['local_epochs']
        self.max_grad_norm = config['training']['max_grad_norm']
        
        print(f"Client {client_id} initialized successfully!")
        print(f"  Training samples: {self.data_module.get_num_samples()}")
        print(f"  Local epochs: {self.local_epochs}")
        print(f"  DP enabled: {self.dp is not None}\n")
    
    def get_parameters(self, config: Dict[str, any]) -> List[np.ndarray]:
        """Return current model parameters as numpy arrays"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: List[np.ndarray]):
        """Set model parameters from numpy arrays"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, any]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train model locally
        
        Args:
            parameters: Global model parameters from server
            config: Training configuration from server
        
        Returns:
            parameters: Updated local model parameters
            num_examples: Number of training samples
            metrics: Training metrics
        """
        print(f"\n[Client {self.client_id}] Starting local training...")
        
        # Set global model parameters
        self.set_parameters(parameters)

        # Snapshot the global params this round started from - DP is applied
        # once at the end of train() to the resulting update, not per-batch.
        global_params = [p.clone().detach() for p in self.model.parameters()]

        # Train locally
        train_loss, train_metrics = self.train(global_params=global_params)
        
        # Get updated parameters
        updated_parameters = self.get_parameters(config={})
        
        # Number of training samples
        num_examples = self.data_module.get_num_samples()
        
        # Metrics to return
        metrics = {
            'train_loss': train_loss,
            'train_accuracy': train_metrics['accuracy'],
            'client_id': self.client_id
        }
        
        print(f"[Client {self.client_id}] Local training complete!")
        print(f"  Loss: {train_loss:.4f}")
        print(f"  Accuracy: {train_metrics['accuracy']:.4f}\n")
        
        return updated_parameters, num_examples, metrics
    
    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, any]
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate model on local validation data
        
        Args:
            parameters: Model parameters to evaluate
            config: Evaluation configuration
        
        Returns:
            loss: Validation loss
            num_examples: Number of validation samples
            metrics: Evaluation metrics
        """
        # Set parameters
        self.set_parameters(parameters)
        
        # Evaluate
        val_loss, val_metrics = self.validate()
        
        num_examples = len(self.data_module.val_dataset)
        
        metrics = {
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'client_id': self.client_id
        }
        
        return val_loss, num_examples, metrics
    
    def train(self, global_params=None) -> Tuple[float, Dict]:
        """
        Local training loop

        Args:
            global_params: model parameters at the start of the round
                (needed to apply client-level DP to the resulting update
                once training finishes - see dp_utils.DifferentialPrivacy)

        Returns:
            avg_loss: Average training loss
            metrics: Training metrics
        """
        self.model.train()
        train_loader = self.data_module.train_dataloader()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            
            for batch in train_loader:
                sequences = batch['sequences'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(sequences, graph_features=None, labels=labels)
                
                # Compute loss
                loss, loss_dict = self.model.compute_loss(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for training stability (not the DP
                # mechanism - see below, DP is applied once per round to
                # the whole local update, not per-batch to raw gradients).
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )

                # Optimizer step
                self.optimizer.step()

                # Metrics
                epoch_loss += loss.item()
                predictions = (outputs['fraud_prob'] > 0.5).long()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

            total_loss += epoch_loss / len(train_loader)

        # Apply DP once per round: clip + noise the whole local update.
        if self.dp is not None and global_params is not None:
            self.dp.privatize_update(self.model, global_params)

        avg_loss = total_loss / self.local_epochs
        accuracy = correct / total if total > 0 else 0.0

        metrics = {'accuracy': accuracy}

        return avg_loss, metrics
    
    def validate(self) -> Tuple[float, Dict]:
        """
        Validation loop
        
        Returns:
            avg_loss: Average validation loss
            metrics: Validation metrics (accuracy, precision, recall, F1)
        """
        self.model.eval()
        val_loader = self.data_module.val_dataloader()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                sequences = batch['sequences'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(sequences, graph_features=None, labels=labels)
                
                # Compute loss
                loss, _ = self.model.compute_loss(outputs, labels)
                total_loss += loss.item()
                
                # Predictions
                predictions = (outputs['fraud_prob'] > 0.5).long()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        
        # Compute metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(all_labels, all_predictions),
            'precision': precision_score(all_labels, all_predictions, zero_division=0),
            'recall': recall_score(all_labels, all_predictions, zero_division=0),
            'f1': f1_score(all_labels, all_predictions, zero_division=0)
        }
        
        return avg_loss, metrics


def create_client_fn(config: dict, device: str = 'cpu'):
    """
    Factory function to create FL clients
    
    Args:
        config: Experiment configuration
        device: 'cpu' or 'cuda'
    
    Returns:
        client_fn: Function that creates a client given a client_id
    """
    def client_fn(cid: str) -> FraudDetectionClient:
        """Create a Flower client for given client ID"""
        client_id = int(cid)
        return FraudDetectionClient(client_id, config, device)
    
    return client_fn


if __name__ == "__main__":
    print("Testing FL Client...")
    
    # This would normally be tested with actual Flower server
    print("FL Client implementation complete!")
    print("Run with Flower server to test full FL workflow.")
