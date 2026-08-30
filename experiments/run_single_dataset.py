import sys
import os
sys.path.append(os.path.abspath('..'))  # Crucial to fix Ray module error for 'src'

import argparse
import yaml
import json
import torch
from pathlib import Path
from datetime import datetime

from train_federated import run_simulation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['paysim', 'ieee_cis', 'european_cc_2013', 'sparkov'])
    parser.add_argument('--rounds', type=int, default=3, 
                        help='Number of federated learning rounds')
    parser.add_argument('--fraction', type=float, default=0.2, 
                        help='Fraction of clients to use per round')
    parser.add_argument('--num-clients', type=int, default=10, 
                        help='Number of clients to simulate (shorter clusters)')
    args = parser.parse_args()

    print("="*80)
    print(f"🚀 LAUNCHING ACTUAL FL SIMULATION FOR: {args.dataset.upper()}")
    print("="*80)

    base_data_dir = "../../datasets_capstone/preprocessed_datasets_final"
    results_dir = Path('../results')
    results_dir.mkdir(exist_ok=True)
    
    with open('../configs/experiment_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    dataset_path = f"{base_data_dir}/{args.dataset}"
    metadata_path = f"{dataset_path}/metadata/dataset_info.json"
    
    if not os.path.exists(metadata_path):
        print(f"❌ Error: Dataset metadata not found at {metadata_path}")
        return

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    config['dataset']['name'] = args.dataset
    config['dataset']['data_path'] = dataset_path
    config['dataset']['num_features'] = metadata['num_features']
    
    # Configure simulation based on passed arguments
    config['federated']['fraction_fit'] = args.fraction
    config['federated']['num_rounds'] = args.rounds
    config['dataset']['num_clients'] = args.num_clients
    config['privacy']['enabled'] = True
    config['privacy']['epsilon'] = 5.0
    # Leverage Mac's MPS GPU accelerator to prevent CPU out-of-memory exhaustion
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # Start Ray-based FL Simulation
        history = run_simulation(config, device=device)
        
        # Save actual empirical history mapped to this dataset
        import pickle
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = results_dir / f"actual_simulation_{args.dataset}_{timestamp}.pkl"
        
        with open(out_file, 'wb') as f:
            pickle.dump(history, f)
            
        print(f"\n✅ Finished {args.dataset.upper()}! History saved to {out_file}")
        
    except Exception as e:
        print(f"❌ Critical Error during FL Simulation: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        import ray
        if ray.is_initialized():
            print("Cleaning up Ray session...")
            ray.shutdown()

if __name__ == "__main__":
    main()
