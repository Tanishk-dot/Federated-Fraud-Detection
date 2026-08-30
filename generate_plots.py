#!/usr/bin/env python3
"""
Generate plots and analysis from real FL training results.
Creates: convergence curves, confusion matrix, ROC curve, client comparison, etc.
All plots use REAL data from training_history.json — nothing fabricated.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.metrics import roc_curve, auc


def load_history(results_dir):
    """Load training history from JSON."""
    history_path = Path(results_dir) / 'training_history.json'
    if not history_path.exists():
        print(f"ERROR: {history_path} not found")
        sys.exit(1)
    with open(history_path) as f:
        return json.load(f)


def plot_convergence(history, save_dir):
    """Plot accuracy, loss, F1, and AUC over FL rounds."""
    rounds = history['rounds']
    metrics = history['global_metrics']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Federated Learning Convergence — Real Training Results', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Accuracy
    acc = [m['accuracy'] for m in metrics]
    axes[0, 0].plot(rounds, acc, 'b-o', linewidth=2, markersize=4, label='Global Accuracy')
    axes[0, 0].set_xlabel('FL Round')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title(f'Accuracy (Final: {acc[-1]:.4f})')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1.05])
    
    # Loss
    loss = [m['loss'] for m in metrics]
    axes[0, 1].plot(rounds, loss, 'r-o', linewidth=2, markersize=4, label='Global Loss')
    axes[0, 1].set_xlabel('FL Round')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title(f'Loss (Final: {loss[-1]:.4f})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    f1 = [m['f1'] for m in metrics]
    axes[1, 0].plot(rounds, f1, 'g-o', linewidth=2, markersize=4, label='F1-Score')
    axes[1, 0].set_xlabel('FL Round')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].set_title(f'F1-Score (Final: {f1[-1]:.4f})')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1.05])
    
    # ROC-AUC
    roc_auc = [m['roc_auc'] for m in metrics]
    axes[1, 1].plot(rounds, roc_auc, 'm-o', linewidth=2, markersize=4, label='ROC-AUC')
    axes[1, 1].set_xlabel('FL Round')
    axes[1, 1].set_ylabel('ROC-AUC')
    axes[1, 1].set_title(f'ROC-AUC (Final: {roc_auc[-1]:.4f})')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1.05])
    
    plt.tight_layout()
    save_path = Path(save_dir) / 'convergence_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_precision_recall(history, save_dir):
    """Plot Precision vs Recall over rounds."""
    rounds = history['rounds']
    metrics = history['global_metrics']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    prec = [m['precision'] for m in metrics]
    rec = [m['recall'] for m in metrics]
    
    ax.plot(rounds, prec, 'b-o', linewidth=2, markersize=5, label='Precision')
    ax.plot(rounds, rec, 'r-o', linewidth=2, markersize=5, label='Recall')
    ax.fill_between(rounds, prec, rec, alpha=0.1, color='purple')
    
    ax.set_xlabel('FL Round', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Precision & Recall Over FL Rounds\n'
                 f'Final: Precision={prec[-1]:.4f}, Recall={rec[-1]:.4f}',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    save_path = Path(save_dir) / 'precision_recall_curve.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrix(history, save_dir):
    """Plot confusion matrix from final round."""
    final = history['global_metrics'][-1]
    cm = np.array(final['confusion_matrix'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    classes = ['Legitimate', 'Fraud']
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=classes, yticklabels=classes,
           xlabel='Predicted', ylabel='Actual',
           title=f'Confusion Matrix (Final Round)\n'
                 f'Accuracy={final["accuracy"]:.4f}')
    
    # Text annotations
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=16, fontweight='bold')
    
    # Add labels
    tn, fp, fn, tp = cm.ravel()
    ax.text(0, -0.3, f'TN={tn}', ha='center', fontsize=10, color='green')
    ax.text(1, -0.3, f'FP={fp}', ha='center', fontsize=10, color='red')
    ax.text(0, 2.3, f'FN={fn}', ha='center', fontsize=10, color='red')
    ax.text(1, 2.3, f'TP={tp}', ha='center', fontsize=10, color='green')
    
    plt.tight_layout()
    save_path = Path(save_dir) / 'confusion_matrix.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_fpr_over_rounds(history, save_dir):
    """Plot False Positive Rate reduction over rounds."""
    rounds = history['rounds']
    metrics = history['global_metrics']
    
    fpr = [m['fpr'] for m in metrics]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(rounds, fpr, alpha=0.3, color='red')
    ax.plot(rounds, fpr, 'r-o', linewidth=2, markersize=5, label='FPR')
    ax.axhline(y=0.01, color='green', linestyle='--', linewidth=1, label='Target (1%)')
    
    ax.set_xlabel('FL Round', fontsize=12)
    ax.set_ylabel('False Positive Rate', fontsize=12)
    ax.set_title(f'False Positive Rate Over FL Rounds\nFinal FPR: {fpr[-1]:.4f}',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = Path(save_dir) / 'fpr_over_rounds.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_client_comparison(history, save_dir):
    """Plot per-client accuracy from the last round."""
    if not history['client_metrics']:
        return
    
    # Get last round's client metrics
    last_round = history['client_metrics'][-1]
    
    client_ids = [c['client_id'] for c in last_round]
    val_acc = [c['val_accuracy'] for c in last_round]
    val_f1 = [c['val_f1'] for c in last_round]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(client_ids))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, val_acc, width, label='Accuracy', color='steelblue')
    bars2 = ax.bar(x + width/2, val_f1, width, label='F1-Score', color='coral')
    
    ax.set_xlabel('Client (Bank)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Client Performance (Final Round)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Bank {c}' for c in client_ids])
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom',
                    fontsize=8)
    
    plt.tight_layout()
    save_path = Path(save_dir) / 'client_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_summary(history, save_dir):
    """Create a summary table as an image."""
    final = history['global_metrics'][-1]
    config = history['config']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    title = 'Federated Learning — Training Summary'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Configuration table
    config_data = [
        ['Dataset', config['dataset']],
        ['FL Rounds', str(config['num_rounds'])],
        ['Total Clients', str(config['num_clients'])],
        ['Clients/Round', str(config['clients_per_round'])],
        ['Local Epochs', str(config['local_epochs'])],
        ['Batch Size', str(config['batch_size'])],
        ['Learning Rate', str(config['learning_rate'])],
        ['FedProx μ', str(config['mu'])],
        ['DP Enabled', str(config['dp_enabled'])],
        ['Model Parameters', f"{config['total_params']:,}"],
    ]
    
    if config.get('dp_enabled'):
        config_data.append(['Noise Multiplier', str(config.get('noise_multiplier', 'N/A'))])
        config_data.append(['Clip Norm', str(config.get('clip_norm', 'N/A'))])
    
    # Results table
    results_data = [
        ['Accuracy', f"{final['accuracy']:.4f}"],
        ['Precision', f"{final['precision']:.4f}"],
        ['Recall', f"{final['recall']:.4f}"],
        ['F1-Score', f"{final['f1']:.4f}"],
        ['ROC-AUC', f"{final['roc_auc']:.4f}"],
        ['PR-AUC', f"{final['pr_auc']:.4f}"],
        ['FPR', f"{final['fpr']:.4f}"],
    ]
    
    # Draw both tables
    table1 = ax.table(cellText=config_data, colLabels=['Configuration', 'Value'],
                      loc='center left', cellLoc='left',
                      bbox=[0.0, 0.0, 0.45, 1.0])
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1, 1.5)
    
    table2 = ax.table(cellText=results_data, colLabels=['Metric', 'Value'],
                      loc='center right', cellLoc='left',
                      bbox=[0.55, 0.3, 0.45, 0.7])
    table2.auto_set_font_size(False)
    table2.set_fontsize(11)
    table2.scale(1, 1.5)
    
    # Color headers
    for key, cell in table1.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')
    for key, cell in table2.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor('#2E7D32')
            cell.set_text_props(color='white', fontweight='bold')
    
    plt.tight_layout()
    save_path = Path(save_dir) / 'training_summary.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def generate_markdown_report(history, save_dir):
    """Generate a markdown report with all results."""
    final = history['global_metrics'][-1]
    config = history['config']
    rounds = history['rounds']
    metrics = history['global_metrics']
    
    report = f"""# Federated Learning Training Results

## Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | {config['dataset']} |
| FL Rounds | {config['num_rounds']} |
| Clients | {config['num_clients']} |
| Clients/Round | {config['clients_per_round']} |
| Local Epochs | {config['local_epochs']} |
| Batch Size | {config['batch_size']} |
| Learning Rate | {config['learning_rate']} |
| FedProx μ | {config['mu']} |
| DP Enabled | {config['dp_enabled']} |
| Model Params | {config['total_params']:,} |

## Final Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | **{final['accuracy']:.4f}** |
| Precision | {final['precision']:.4f} |
| Recall | {final['recall']:.4f} |
| **F1-Score** | **{final['f1']:.4f}** |
| **ROC-AUC** | **{final['roc_auc']:.4f}** |
| PR-AUC | {final['pr_auc']:.4f} |
| FPR | {final['fpr']:.4f} |

## Confusion Matrix

|  | Predicted Legit | Predicted Fraud |
|--|-----------------|-----------------|
| **Actual Legit** | TN={final['tn']} | FP={final['fp']} |
| **Actual Fraud** | FN={final['fn']} | TP={final['tp']} |

## Convergence Over Rounds

| Round | Accuracy | F1 | ROC-AUC | FPR |
|-------|----------|----|---------|-----|
"""
    for i, r in enumerate(rounds):
        m = metrics[i]
        report += f"| {r} | {m['accuracy']:.4f} | {m['f1']:.4f} | {m['roc_auc']:.4f} | {m['fpr']:.4f} |\n"
    
    report += """
## Plots

- ![Convergence](convergence_curves.png)
- ![Confusion Matrix](confusion_matrix.png)
- ![Precision & Recall](precision_recall_curve.png)
- ![FPR Over Rounds](fpr_over_rounds.png)
- ![Client Comparison](client_comparison.png)
"""
    
    save_path = Path(save_dir) / 'TRAINING_RESULTS.md'
    with open(save_path, 'w') as f:
        f.write(report)
    print(f"  Saved: {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, required=True)
    args = parser.parse_args()
    
    print(f"\nGenerating plots from: {args.results_dir}")
    print("=" * 50)
    
    history = load_history(args.results_dir)
    
    print(f"Dataset: {history['config']['dataset']}")
    print(f"Rounds: {len(history['rounds'])}")
    final = history['global_metrics'][-1]
    print(f"Final accuracy: {final['accuracy']:.4f}")
    print(f"Final F1: {final['f1']:.4f}")
    print(f"Final AUC: {final['roc_auc']:.4f}")
    print()
    
    plot_convergence(history, args.results_dir)
    plot_precision_recall(history, args.results_dir)
    plot_confusion_matrix(history, args.results_dir)
    plot_fpr_over_rounds(history, args.results_dir)
    plot_client_comparison(history, args.results_dir)
    plot_training_summary(history, args.results_dir)
    generate_markdown_report(history, args.results_dir)
    
    print("\n✅ All plots generated successfully!")


if __name__ == '__main__':
    main()
