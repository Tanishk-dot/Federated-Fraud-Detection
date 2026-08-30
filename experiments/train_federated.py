"""
Main Federated Learning Training Script
Runs FL simulation with Flower framework
"""

import sys
sys.path.append('..')

import yaml
import torch
import flwr as fl
from pathlib import Path
import argparse
import numpy as np

from src.federated.client import create_client_fn
from src.models.temporal_graph_transformer import TemporalGraphTransformer


def load_config(config_path='../configs/experiment_config.yaml'):
    """Load experiment configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def weighted_average(metrics):
    """Aggregate metrics from multiple clients"""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["val_accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Aggregate and return custom metric (weighted average)
    return {"val_accuracy": sum(accuracies) / sum(examples)}


def get_evaluate_fn(config, device):
    """
    Return an evaluation function for server-side evaluation
    """
    # Load test data from client 0 (or aggregate from all clients)
    from src.data.dataset import FraudDetectionDataModule
    
    data_module = FraudDetectionDataModule(config, client_id=0)
    data_module.setup()
    test_loader = data_module.test_dataloader()
    
    # Create model
    model = TemporalGraphTransformer(config).to(device)
    
    def evaluate(server_round, parameters, config_dict):
        """
        Evaluate global model on centralized test set
        """
        # Set model parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        from collections import OrderedDict
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        
        # Evaluate
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_fraud_probs = []
        
        with torch.no_grad():
            for batch in test_loader:
                sequences = batch['sequences'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(sequences, graph_features=None, labels=labels)
                loss, _ = model.compute_loss(outputs, labels)
                
                total_loss += loss.item()
                predictions = (outputs['fraud_prob'] > 0.5).long()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_fraud_probs.extend(outputs['fraud_prob'].cpu().numpy())
        
        avg_loss = total_loss / len(test_loader)
        
        # Compute metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score
        )
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy_score(all_labels, all_predictions),
            'precision': precision_score(all_labels, all_predictions, zero_division=0),
            'recall': recall_score(all_labels, all_predictions, zero_division=0),
            'f1': f1_score(all_labels, all_predictions, zero_division=0),
            'roc_auc': roc_auc_score(all_labels, all_fraud_probs) if len(set(all_labels)) > 1 else 0.0
        }
        
        print(f"\n{'='*60}")
        print(f"Round {server_round} - Global Model Evaluation")
        print(f"{'='*60}")
        print(f"Loss: {metrics['loss']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"{'='*60}\n")
        
        return avg_loss, metrics
    
    return evaluate


def run_simulation(config, device='cpu'):
    """
    Run FL simulation (all clients in one process)
    """
    print("\n" + "="*60)
    print("FEDERATED FRAUD DETECTION - SIMULATION MODE")
    print("="*60 + "\n")
    
    # Create client function
    client_fn = create_client_fn(config, device)
    
    # Create FedProx strategy
    strategy = fl.server.strategy.FedProx(
        fraction_fit=config['federated']['fraction_fit'],
        fraction_evaluate=config['federated']['fraction_fit'],
        min_fit_clients=config['federated']['min_fit_clients'],
        min_evaluate_clients=config['federated']['min_fit_clients'],
        min_available_clients=config['federated']['min_available_clients'],
        evaluate_metrics_aggregation_fn=weighted_average,
        evaluate_fn=get_evaluate_fn(config, device),
        proximal_mu=config['federated']['proximal_mu'],
    )
    
    # Specify client resources (MPS passes logic as CPU because Ray does not index Apple Silicon as native CUDA slots)
    client_resources = {
        "num_cpus": 1,
        "num_gpus": 0.1 if device == 'cuda' else 0.0
    }
    
    print(f"Starting FL simulation with {config['dataset']['num_clients']} clients...")
    print(f"Strategy: FedProx (μ={config['federated']['proximal_mu']})")
    print(f"Rounds: {config['federated']['num_rounds']}")
    print(f"Fraction fit: {config['federated']['fraction_fit']}")
    print(f"Device: {device}\n")
    
    import os
    ray_init_args = {
        "_temp_dir": "/Volumes/RayData",  # APFS Virtual disk physically stored on Untitled
        "object_store_memory": 2 * 1024 * 1024 * 1024, # 2 GB limit
        "runtime_env": {"env_vars": {"PYTHONPATH": os.path.abspath('..')}}
    }
    
    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config['dataset']['num_clients'],
        config=fl.server.ServerConfig(num_rounds=config['federated']['num_rounds']),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args=ray_init_args,
    )
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE!")
    print("="*60 + "\n")
    
    # Print final results
    if history.metrics_centralized:
        final_metrics = history.metrics_centralized['accuracy'][-1]
        print("Final Global Model Performance:")
        print(f"  Round: {final_metrics[0]}")
        print(f"  Accuracy: {final_metrics[1]:.4f}")
    
    return history


def main():
    parser = argparse.ArgumentParser(description='Federated Fraud Detection Training')
    parser.add_argument('--config', type=str, default='../configs/experiment_config.yaml',
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to use')
    parser.add_argument('--mode', type=str, default='simulation',
                       choices=['simulation', 'server', 'client'],
                       help='Run mode: simulation (all-in-one), server, or client')
    parser.add_argument('--client-id', type=int, default=0,
                       help='Client ID (for client mode)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    if args.mode == 'simulation':
        # Run full simulation
        history = run_simulation(config, args.device)
        
        # Save results
        results_dir = Path('../results')
        results_dir.mkdir(exist_ok=True)
        
        import pickle
        with open(results_dir / 'fl_history.pkl', 'wb') as f:
            pickle.dump(history, f)
        
        print(f"\nResults saved to {results_dir / 'fl_history.pkl'}")
    
    elif args.mode == 'server':
        print("Server mode not yet implemented")
        print("Use 'simulation' mode for now")
    
    elif args.mode == 'client':
        print("Client mode not yet implemented")
        print("Use 'simulation' mode for now")


if __name__ == "__main__":
    main()
