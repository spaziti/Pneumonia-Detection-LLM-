"""
Phase 4a: Evaluation script for Pneumonia Detection models.

Computes:
- Accuracy, Precision, Recall, F1-score
- AUC-ROC
- Confusion matrix heatmap
- ROC curve plot
- Training history plots (loss/accuracy curves)
- Model comparison charts

Usage:
    python evaluate.py                    # Evaluate HATR model
    python evaluate.py --model cnn        # Evaluate CNN baseline
    python evaluate.py --model all        # Evaluate all models
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess import get_dataloaders, CLASS_NAMES
from model import build_model

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / ".tmp" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / ".tmp" / "results"
LOG_DIR = PROJECT_ROOT / ".tmp" / "logs"

# Style configuration
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {
    'cnn': '#FF6B6B',
    'vit': '#4ECDC4',
    'hatr': '#45B7D1',
}


def tta_predict(model, images, multimodal=False, tabular=None):
    """
    Run Test-Time Augmentation (TTA) on a batch of images.
    Applies original, horizontal flip, and rotation (+/- 5 degrees) transformations,
    then averages the predicted probabilities.
    """
    # 1. Original
    if multimodal:
        outputs = model(images, tabular)
    else:
        outputs = model(images)
    probs = torch.softmax(outputs, dim=1)
    
    # 2. Horizontal flip
    flipped_images = TF.hflip(images)
    if multimodal:
        flipped_outputs = model(flipped_images, tabular)
    else:
        flipped_outputs = model(flipped_images)
    probs += torch.softmax(flipped_outputs, dim=1)
    
    # 3. Rotate +5 degrees
    rot_pos_images = TF.rotate(images, angle=5)
    if multimodal:
        rot_pos_outputs = model(rot_pos_images, tabular)
    else:
        rot_pos_outputs = model(rot_pos_images)
    probs += torch.softmax(rot_pos_outputs, dim=1)
    
    # 4. Rotate -5 degrees
    rot_neg_images = TF.rotate(images, angle=-5)
    if multimodal:
        rot_neg_outputs = model(rot_neg_images, tabular)
    else:
        rot_neg_outputs = model(rot_neg_images)
    probs += torch.softmax(rot_neg_outputs, dim=1)
    
    # Average the probabilities
    probs /= 4.0
    return probs


@torch.no_grad()
def get_predictions(model, loader, device, multimodal=False, tta=False):
    """Get predictions and probabilities for the entire dataset."""
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    for batch_idx, batch in enumerate(loader):
        if os.environ.get("DRY_RUN") == "1" and batch_idx >= 5:
            break
        if multimodal:
            images, tabular, labels = batch
            images, tabular = images.to(device), tabular.to(device)
            if tta:
                probs = tta_predict(model, images, multimodal=True, tabular=tabular)
            else:
                outputs = model(images, tabular)
                probs = torch.softmax(outputs, dim=1)
        else:
            images, labels = batch
            images = images.to(device)
            if tta:
                probs = tta_predict(model, images, multimodal=False)
            else:
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)

        _, predicted = probs.max(1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # Prob of pneumonia

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_metrics(labels, preds, probs):
    """Compute all evaluation metrics."""
    metrics = {
        'accuracy': accuracy_score(labels, preds) * 100,
        'precision': precision_score(labels, preds, average='binary') * 100,
        'recall': recall_score(labels, preds, average='binary') * 100,
        'f1': f1_score(labels, preds, average='binary') * 100,
        'auc_roc': roc_auc_score(labels, probs) * 100,
    }
    return metrics


def plot_confusion_matrix(labels, preds, model_type, save_dir):
    """Generate and save confusion matrix heatmap."""
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        annot_kws={'size': 16}, linewidths=0.5,
        ax=ax
    )
    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('Actual', fontsize=14)
    ax.set_title(f'Confusion Matrix — {model_type.upper()} Model', fontsize=16, fontweight='bold')

    save_path = save_dir / f'confusion_matrix_{model_type}.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path.name}")


def plot_roc_curve(labels, probs, model_type, save_dir, all_results=None):
    """Generate and save ROC curve."""
    fig, ax = plt.subplots(figsize=(8, 6))

    if all_results:
        # Plot all models for comparison
        for name, result in all_results.items():
            fpr, tpr, _ = roc_curve(result['labels'], result['probs'])
            auc = roc_auc_score(result['labels'], result['probs'])
            ax.plot(fpr, tpr, label=f"{name.upper()} (AUC={auc:.3f})",
                    color=COLORS.get(name, '#333'), linewidth=2)
    else:
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        ax.plot(fpr, tpr, label=f"{model_type.upper()} (AUC={auc:.3f})",
                color=COLORS.get(model_type, '#45B7D1'), linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curve — Pneumonia Detection', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    suffix = "comparison" if all_results else model_type
    save_path = save_dir / f'roc_curve_{suffix}.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path.name}")


def plot_training_history(model_type, save_dir):
    """Plot training loss and accuracy curves from saved history."""
    history_path = LOG_DIR / f"history_{model_type}.json"
    if not history_path.exists():
        print(f"    No training history found for {model_type}")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'o-', label='Train Loss',
             color='#FF6B6B', linewidth=2, markersize=4)
    ax1.plot(epochs, history['val_loss'], 's-', label='Val Loss',
             color='#45B7D1', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'Training Loss — {model_type.upper()}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'o-', label='Train Acc',
             color='#FF6B6B', linewidth=2, markersize=4)
    ax2.plot(epochs, history['val_acc'], 's-', label='Val Acc',
             color='#45B7D1', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title(f'Training Accuracy — {model_type.upper()}', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Mark best epoch
    best_epoch = history.get('best_epoch', 0)
    if best_epoch > 0:
        ax2.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5, label=f'Best (ep {best_epoch})')
        ax2.legend(fontsize=11)

    save_path = save_dir / f'training_history_{model_type}.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path.name}")


def plot_model_comparison(all_metrics, save_dir):
    """Bar chart comparing all model metrics."""
    if len(all_metrics) < 2:
        return

    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']

    x = np.arange(len(metric_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (name, metrics) in enumerate(all_metrics.items()):
        values = [metrics[m] for m in metric_names]
        offset = (i - len(all_metrics)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=name.upper(),
                      color=COLORS.get(name, '#333'), alpha=0.85)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Score (%)', fontsize=13)
    ax.set_title('Model Comparison — All Metrics', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.legend(fontsize=12)
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)

    save_path = save_dir / 'model_comparison.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path.name}")


def evaluate_model(model_type, device, multimodal=False, backbone_type="resnet18", tta=False):
    """Full evaluation of a single model."""
    # First try loading specific checkpoint
    checkpoint_path = CHECKPOINT_DIR / f"best_{model_type}_{backbone_type}.pth"
    if not checkpoint_path.exists():
        # Fallback to default
        checkpoint_path = CHECKPOINT_DIR / f"best_{model_type}.pth"

    if not checkpoint_path.exists():
        print(f"\n  No checkpoint found for {model_type} at {checkpoint_path}")
        return None, None

    print(f"\n{'-' * 50}")
    print(f"  Evaluating: {model_type.upper()} Model")
    print(f"{'-' * 50}")

    # Load checkpoint first to detect backbone type
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actual_backbone = checkpoint.get('backbone_type', backbone_type)
    
    # Load model
    model = build_model(model_type, num_classes=2, pretrained=False,
                         multimodal=multimodal, backbone_type=actual_backbone).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"    Loaded checkpoint from epoch {checkpoint['epoch']} (Backbone: {actual_backbone})")

    # DataLoader
    num_workers = 0 if os.name == 'nt' else 2
    _, _, test_loader = get_dataloaders(batch_size=32, num_workers=num_workers,
                                         multimodal=multimodal)

    # Get predictions
    labels, preds, probs = get_predictions(model, test_loader, device,
                                            multimodal=multimodal, tta=tta)

    # Compute metrics
    metrics = compute_metrics(labels, preds, probs)

    print(f"\n    Test Results:")
    print(f"    {'Accuracy:':<12} {metrics['accuracy']:.2f}%")
    print(f"    {'Precision:':<12} {metrics['precision']:.2f}%")
    print(f"    {'Recall:':<12} {metrics['recall']:.2f}%")
    print(f"    {'F1-Score:':<12} {metrics['f1']:.2f}%")
    print(f"    {'AUC-ROC:':<12} {metrics['auc_roc']:.2f}%")

    # Classification report
    print(f"\n    Classification Report:")
    report = classification_report(labels, preds, target_names=CLASS_NAMES, digits=4)
    for line in report.split('\n'):
        print(f"      {line}")

    # Generate plots
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n    Generating plots...")
    plot_confusion_matrix(labels, preds, model_type, RESULTS_DIR)
    plot_roc_curve(labels, probs, model_type, RESULTS_DIR)
    plot_training_history(model_type, RESULTS_DIR)

    # Save metrics to JSON
    metrics_path = RESULTS_DIR / f'metrics_{model_type}.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"    Saved: {metrics_path.name}")

    return metrics, {'labels': labels, 'preds': preds, 'probs': probs}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Pneumonia Detection Models")
    parser.add_argument('--model', type=str, default='hatr',
                        choices=['cnn', 'vit', 'hatr', 'all'],
                        help='Model type to evaluate (default: hatr)')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet50'],
                        help='CNN backbone type (default: resnet18)')
    parser.add_argument('--tta', action='store_true',
                        help='Apply Test-Time Augmentation (TTA)')
    parser.add_argument('--multimodal', action='store_true',
                        help='Evaluate multi-modal model')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 60}")
    print(f"PHASE 4: Model Evaluation")
    print(f"{'=' * 60}")
    print(f"  Device: {device}")

    models_to_eval = ['cnn', 'vit', 'hatr'] if args.model == 'all' else [args.model]
    all_metrics = {}
    all_results = {}

    for model_type in models_to_eval:
        metrics, results = evaluate_model(
            model_type, device, multimodal=args.multimodal,
            backbone_type=args.backbone, tta=args.tta
        )
        if metrics is not None:
            all_metrics[model_type] = metrics
            all_results[model_type] = results

    # Comparison plots (if multiple models)
    if len(all_metrics) > 1:
        print(f"\n{'-' * 50}")
        print("  Generating comparison plots...")
        plot_roc_curve(None, None, None, RESULTS_DIR, all_results)
        plot_model_comparison(all_metrics, RESULTS_DIR)

        # Summary table
        print(f"\n{'=' * 60}")
        print("  MODEL COMPARISON SUMMARY")
        print(f"  {'Model':<8} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AUC':>8}")
        print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for name, m in all_metrics.items():
            print(f"  {name.upper():<8} {m['accuracy']:>7.2f}% {m['precision']:>7.2f}% "
                  f"{m['recall']:>7.2f}% {m['f1']:>7.2f}% {m['auc_roc']:>7.2f}%")

    print(f"\n{'=' * 60}")
    print("Evaluation complete! Results saved to .tmp/results/")


if __name__ == "__main__":
    main()
