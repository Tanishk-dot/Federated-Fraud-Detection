"""
Dataset loader for preprocessed fraud detection data
Loads PyTorch tensors from preprocessed_datasets_final/
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json


class FraudDetectionDataset(Dataset):
    """
    Dataset for preprocessed fraud detection data
    
    Loads temporal sequences and labels from .pt files
    """
    
    def __init__(self, data_path, split='train'):
        """
        Args:
            data_path: Path to client data (e.g., 'paysim/client_0/')
            split: 'train', 'val', or 'test'
        """
        self.data_path = Path(data_path)
        self.split = split
        
        # Load data
        data_file = self.data_path / f'{split}_temporal.pt'
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        data = torch.load(data_file)
        self.sequences = data['sequences']  # (N, seq_len, features)
        self.labels = data['labels']  # (N,)
        
        print(f"Loaded {split} data: {len(self)} samples")
        print(f"  Sequences shape: {self.sequences.shape}")
        print(f"  Labels shape: {self.labels.shape}")
        print(f"  Fraud rate: {self.labels.float().mean():.2%}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'sequences': self.sequences[idx],  # (seq_len, features)
            'labels': self.labels[idx]  # scalar
        }


class FraudDetectionDataModule:
    """
    Data module for managing train/val/test dataloaders
    """
    
    def __init__(self, config, client_id=0):
        """
        Args:
            config: experiment configuration dict
            client_id: which FL client (0-9)
        """
        self.config = config
        self.client_id = client_id
        
        # Build data path
        dataset_name = config['dataset']['name']
        data_root = Path(config['dataset']['data_path'])
        self.client_path = data_root / f'client_{client_id}'
        
        if not self.client_path.exists():
            raise FileNotFoundError(f"Client data not found: {self.client_path}")
        
        # Load metadata
        metadata_path = data_root / 'metadata' / 'dataset_info.json'
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        print(f"\n{'='*60}")
        print(f"Initializing Data Module for Client {client_id}")
        print(f"Dataset: {dataset_name}")
        print(f"Client path: {self.client_path}")
        print(f"{'='*60}\n")
    
    def setup(self):
        """Load datasets"""
        self.train_dataset = FraudDetectionDataset(self.client_path, 'train')
        self.val_dataset = FraudDetectionDataset(self.client_path, 'val')
        self.test_dataset = FraudDetectionDataset(self.client_path, 'test')
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
    
    def get_num_samples(self):
        """Return number of training samples"""
        return len(self.train_dataset)


if __name__ == "__main__":
    print("Testing Dataset Loader...")
    
    # Mock config
    config = {
        'dataset': {
            'name': 'paysim',
            'data_path': '../datasets_capstone/preprocessed_datasets_final/paysim'
        },
        'training': {
            'batch_size': 32
        }
    }
    
    # Test data module
    try:
        data_module = FraudDetectionDataModule(config, client_id=0)
        data_module.setup()
        
        # Test dataloaders
        train_loader = data_module.train_dataloader()
        batch = next(iter(train_loader))
        
        print(f"\nBatch shapes:")
        print(f"  Sequences: {batch['sequences'].shape}")
        print(f"  Labels: {batch['labels'].shape}")
        
        print("\n✅ Dataset loader test passed!")
        
    except FileNotFoundError as e:
        print(f"\n⚠️ Test skipped: {e}")
        print("This is expected if preprocessed data is not in the expected location.")
