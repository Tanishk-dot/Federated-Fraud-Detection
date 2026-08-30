"""
Test script to verify installation and setup
"""

import sys
import torch
import yaml
from pathlib import Path

def test_imports():
    """Test that all required packages are installed"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"  ✅ PyTorch {torch.__version__}")
    except ImportError:
        print("  ❌ PyTorch not installed")
        return False
    
    try:
        import flwr
        print(f"  ✅ Flower {flwr.__version__}")
    except ImportError:
        print("  ❌ Flower not installed")
        return False
    
    try:
        import numpy
        print(f"  ✅ NumPy {numpy.__version__}")
    except ImportError:
        print("  ❌ NumPy not installed")
        return False
    
    try:
        import sklearn
        print(f"  ✅ scikit-learn {sklearn.__version__}")
    except ImportError:
        print("  ❌ scikit-learn not installed")
        return False
    
    return True


def test_config():
    """Test that configuration file exists and is valid"""
    print("\nTesting configuration...")
    
    config_path = Path('configs/experiment_config.yaml')
    if not config_path.exists():
        print(f"  ❌ Config file not found: {config_path}")
        return False
    
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        print(f"  ✅ Config loaded successfully")
        print(f"     Dataset: {config['dataset']['name']}")
        print(f"     Clients: {config['dataset']['num_clients']}")
        print(f"     Rounds: {config['federated']['num_rounds']}")
        return True
    except Exception as e:
        print(f"  ❌ Error loading config: {e}")
        return False


def test_models():
    """Test that model components can be imported and instantiated"""
    print("\nTesting model components...")
    
    try:
        from src.models.temporal_transformer import TemporalTransformer
        model = TemporalTransformer(input_dim=10, hidden_dim=128)
        print("  ✅ Temporal Transformer")
    except Exception as e:
        print(f"  ❌ Temporal Transformer: {e}")
        return False
    
    try:
        from src.models.graph_encoder import SimpleGraphEncoder
        model = SimpleGraphEncoder(hidden_dim=128)
        print("  ✅ Graph Encoder")
    except Exception as e:
        print(f"  ❌ Graph Encoder: {e}")
        return False
    
    try:
        from src.models.fusion import CrossModalFusion
        model = CrossModalFusion(hidden_dim=128)
        print("  ✅ Cross-Modal Fusion")
    except Exception as e:
        print(f"  ❌ Cross-Modal Fusion: {e}")
        return False
    
    try:
        from src.models.contrastive_head import ContrastiveLearningHead
        model = ContrastiveLearningHead(hidden_dim=128)
        print("  ✅ Contrastive Learning Head")
    except Exception as e:
        print(f"  ❌ Contrastive Learning Head: {e}")
        return False
    
    try:
        with open('configs/experiment_config.yaml') as f:
            config = yaml.safe_load(f)
        from src.models.temporal_graph_transformer import TemporalGraphTransformer
        model = TemporalGraphTransformer(config)
        print("  ✅ Complete Temporal Graph Transformer")
    except Exception as e:
        print(f"  ❌ Complete Model: {e}")
        return False
    
    return True


def test_data_path():
    """Test that data path exists"""
    print("\nTesting data path...")
    
    try:
        with open('configs/experiment_config.yaml') as f:
            config = yaml.safe_load(f)
        
        data_path = Path(config['dataset']['data_path'])
        
        if not data_path.exists():
            print(f"  ⚠️  Data path not found: {data_path}")
            print(f"     This is expected if you haven't set up the data yet.")
            print(f"     Update 'data_path' in configs/experiment_config.yaml")
            return False
        
        # Check for client folders
        client_0 = data_path / 'client_0'
        if not client_0.exists():
            print(f"  ⚠️  Client data not found: {client_0}")
            return False
        
        # Check for data files
        train_file = client_0 / 'train_temporal.pt'
        if not train_file.exists():
            print(f"  ⚠️  Training data not found: {train_file}")
            return False
        
        print(f"  ✅ Data path exists: {data_path}")
        print(f"  ✅ Client 0 data found")
        return True
        
    except Exception as e:
        print(f"  ❌ Error checking data path: {e}")
        return False


def test_forward_pass():
    """Test a complete forward pass through the model"""
    print("\nTesting forward pass...")
    
    try:
        with open('configs/experiment_config.yaml') as f:
            config = yaml.safe_load(f)
        
        from src.models.temporal_graph_transformer import TemporalGraphTransformer
        
        model = TemporalGraphTransformer(config)
        
        # Create dummy input
        batch_size = 4
        seq_len = 10
        num_features = config['dataset']['num_features']
        
        sequences = torch.randn(batch_size, seq_len, num_features)
        labels = torch.randint(0, 2, (batch_size,))
        
        # Forward pass
        model.train()
        outputs = model(sequences, graph_features=None, labels=labels)
        
        # Check outputs
        assert 'fraud_prob' in outputs
        assert 'anomaly_scores' in outputs
        assert outputs['fraud_prob'].shape == (batch_size,)
        
        print("  ✅ Forward pass successful")
        print(f"     Input shape: {sequences.shape}")
        print(f"     Output shape: {outputs['fraud_prob'].shape}")
        return True
        
    except Exception as e:
        print(f"  ❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("FEDERATED FRAUD DETECTION - SETUP TEST")
    print("="*60 + "\n")
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_config()))
    results.append(("Model Components", test_models()))
    results.append(("Data Path", test_data_path()))
    results.append(("Forward Pass", test_forward_pass()))
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60 + "\n")
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:.<40} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nYou're ready to run federated learning:")
        print("  cd experiments")
        print("  python train_federated.py --mode simulation")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("\nPlease fix the issues above before running training.")
        print("Check README.md for setup instructions.")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
