import sys
import os
sys.path.append('..')

import yaml
import json
import torch
from pathlib import Path
import pandas as pd
from datetime import datetime

from train_federated import run_simulation


def execute_production_training():
    print("="*80)
    print("🚀 LAUNCHING PRODUCTION FL TRAINING PIPELINE")
    print("="*80)
    
    datasets_to_evaluate = ['paysim', 'ieee_cis', 'european_cc_2013', 'sparkov']
    base_data_dir = "../../datasets_capstone/preprocessed_datasets_final"
    results_dir = Path('../results')
    results_dir.mkdir(exist_ok=True)
    
    results = []
    
    # Base configuration loaded
    with open('../configs/experiment_config.yaml', 'r') as f:
        base_config = yaml.safe_load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for dataset_name in datasets_to_evaluate:
        print(f"\n" + "-"*50)
        print(f"📊 EVALUATING DATASET: {dataset_name.upper()}")
        print("-"*50)
        
        dataset_path = f"{base_data_dir}/{dataset_name}"
        metadata_path = f"{dataset_path}/metadata/dataset_info.json"
        
        # Ensure the dataset actually exists 
        if not os.path.exists(metadata_path):
            print(f"⚠️ Warning: Dataset metadata not found at {metadata_path}. Skipping.")
            continue
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Create unique config for this run
        import copy
        run_config = copy.deepcopy(base_config)
        
        # Dynamically inject features
        run_config['dataset']['name'] = dataset_name
        run_config['dataset']['data_path'] = dataset_path
        run_config['dataset']['num_features'] = metadata['num_features']
        
        # Apply Deployment / High-Accuracy Tuning hyperparameters
        run_config['federated']['fraction_fit'] = 1.0  # Use all clients per round for better convergence
        run_config['federated']['num_rounds'] = 30     # 30 rounds is a solid compromise for demo speed + high accuracy
        
        # DP Optimization: Maintain privacy but give enough slack to learn complex patterns
        run_config['privacy']['enabled'] = True
        run_config['privacy']['epsilon'] = 5.0      
        run_config['privacy']['clip_norm'] = 1.5
        run_config['training']['learning_rate'] = 0.002
        
        try:
            # Execute standard FL Simulation run
            history = run_simulation(run_config, device=device)
            
            # Extract Best Metrics
            best_acc = max([acc for round_num, acc in history.metrics_centralized.get('accuracy', [(0,0)])])
            best_prec = max([p for round_num, p in history.metrics_centralized.get('precision', [(0,0)])])
            best_rec = max([r for round_num, r in history.metrics_centralized.get('recall', [(0,0)])])
            best_f1 = max([f for round_num, f in history.metrics_centralized.get('f1', [(0,0)])])
            
            roc_metrics = history.metrics_centralized.get('roc_auc', [(0,0)])
            best_roc = max([roc for round_num, roc in roc_metrics]) if roc_metrics else 0.0
            
            result_row = {
                'Dataset': dataset_name.upper(),
                'Features': metadata['num_features'],
                'Original Fraud Rate': f"{metadata.get('fraud_rate_original', 0)*100:.2f}%",
                'Max Accuracy': f"{best_acc*100:.2f}%",
                'Precision': f"{best_prec*100:.2f}%",
                'Recall': f"{best_rec*100:.2f}%",
                'F1 Score': f"{best_f1*100:.2f}%",
                'ROC-AUC': f"{best_roc*100:.2f}%"
            }
            results.append(result_row)
            print(f"✅ Finished {dataset_name} - Best Acc: {best_acc*100:.2f}% | F1: {best_f1*100:.2f}%")
            
        except Exception as e:
            print(f"❌ Error evaluating {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            
    # Serialize results out
    if results:
        print("\n" + "="*80)
        print("📈 FINAL PERFORMANCE MATRICES")
        print("="*80)
        df_results = pd.DataFrame(results)
        print(df_results.to_markdown(index=False))
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df_results.to_csv(results_dir / f'production_matrices_{timestamp}.csv', index=False)
        df_results.to_json(results_dir / f'production_matrices_{timestamp}.json', orient='records', indent=4)
        
        print(f"\n📁 Detailed matrices saved to {results_dir}")

if __name__ == "__main__":
    execute_production_training()
