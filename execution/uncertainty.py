"""
Feature 2: Uncertainty Quantification via Monte Carlo Dropout.

Instead of a single deterministic prediction, runs N stochastic forward passes
with dropout enabled at inference time. The variance across passes yields an
"Uncertainty Score" — high uncertainty flags the case for human review.

Usage:
    python uncertainty.py --model hatr --n-passes 50
    python uncertainty.py --model hatr --n-passes 20 --threshold 0.15
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess import get_dataloaders, CLASS_NAMES
from model import build_model

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / ".tmp" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / ".tmp" / "results"


# ---------------------------------------------------------------------------
# Core MC Dropout Utilities
# ---------------------------------------------------------------------------

def enable_mc_dropout(model):
    """
    Enable dropout layers during inference for Monte Carlo sampling.
    Keeps BatchNorm and other layers in eval mode — only Dropout is toggled.
    """
    model.eval()  # Everything to eval first
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()  # Re-enable dropout


def mc_predict(model, image_tensor, n_passes=50):
    """
    Run *n_passes* stochastic forward passes and aggregate results.

    Args:
        model: PyTorch model (dropout layers must be in train mode)
        image_tensor: (1, 3, H, W) preprocessed input
        n_passes: number of stochastic forward passes

    Returns:
        mean_prob:   float — mean predicted probability of the positive class
        std_prob:    float — standard deviation (= uncertainty)
        all_probs:   np.array of shape (n_passes,) — individual pass probabilities
        pred_class:  int — final predicted class (from mean_prob)
    """
    all_probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            output = model(image_tensor)
            prob = F.softmax(output, dim=1)[:, 1]  # P(Pneumonia)
            all_probs.append(prob.cpu().item())

    all_probs = np.array(all_probs)
    mean_prob = all_probs.mean()
    std_prob = all_probs.std()
    pred_class = 1 if mean_prob >= 0.5 else 0

    return mean_prob, std_prob, all_probs, pred_class


def classify_with_uncertainty(model, dataloader, device, n_passes=50):
    """
    Run MC Dropout inference on an entire dataloader.

    Returns:
        results: list of dicts with keys:
            label, pred, mean_prob, uncertainty, correct
    """
    enable_mc_dropout(model)
    results = []

    pbar = tqdm(dataloader, desc="MC Dropout Inference", ncols=100)
    for images, labels in pbar:
        for i in range(images.size(0)):
            img = images[i:i+1].to(device)
            label = labels[i].item()

            mean_prob, std_prob, _, pred_class = mc_predict(
                model, img, n_passes=n_passes
            )

            results.append({
                'label': label,
                'pred': pred_class,
                'mean_prob': float(mean_prob),
                'uncertainty': float(std_prob),
                'correct': pred_class == label,
            })

    return results


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_uncertainty_distribution(results, save_dir):
    """Histogram of uncertainty scores, split by correct vs incorrect."""
    correct_unc = [r['uncertainty'] for r in results if r['correct']]
    wrong_unc   = [r['uncertainty'] for r in results if not r['correct']]

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, max(max(correct_unc, default=0), max(wrong_unc, default=0)) + 0.02, 30)

    ax.hist(correct_unc, bins=bins, alpha=0.7, label=f'Correct ({len(correct_unc)})',
            color='#2ecc71', edgecolor='white')
    ax.hist(wrong_unc, bins=bins, alpha=0.7, label=f'Incorrect ({len(wrong_unc)})',
            color='#e74c3c', edgecolor='white')

    ax.set_xlabel('Uncertainty (σ)', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title('MC Dropout Uncertainty Distribution', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    save_path = save_dir / 'uncertainty_distribution.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path.name}")


def plot_reliability_diagram(results, save_dir, n_bins=10):
    """Calibration / reliability diagram."""
    probs  = np.array([r['mean_prob'] for r in results])
    labels = np.array([r['label'] for r in results])

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs  = []
    bin_confs = []
    bin_counts = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            bin_accs.append(0)
            bin_confs.append((lo + hi) / 2)
            bin_counts.append(0)
        else:
            bin_accs.append(labels[mask].mean())
            bin_confs.append(probs[mask].mean())
            bin_counts.append(mask.sum())

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.bar(bin_confs, bin_accs, width=1.0 / n_bins * 0.8, alpha=0.7,
           color='#45B7D1', edgecolor='white', label='Model')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')

    ax.set_xlabel('Mean Predicted Probability', fontsize=13)
    ax.set_ylabel('Fraction of Positives', fontsize=13)
    ax.set_title('Reliability Diagram (Calibration)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(alpha=0.3)

    save_path = save_dir / 'reliability_diagram.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path.name}")


def plot_uncertainty_vs_accuracy(results, save_dir, n_bins=8):
    """Show how accuracy changes across uncertainty buckets."""
    sorted_res = sorted(results, key=lambda r: r['uncertainty'])
    chunk = len(sorted_res) // n_bins or 1
    buckets_acc = []
    buckets_unc = []

    for i in range(0, len(sorted_res), chunk):
        bucket = sorted_res[i:i + chunk]
        if len(bucket) == 0:
            continue
        acc = sum(1 for r in bucket if r['correct']) / len(bucket) * 100
        mean_unc = np.mean([r['uncertainty'] for r in bucket])
        buckets_acc.append(acc)
        buckets_unc.append(mean_unc)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(buckets_acc)), buckets_acc, color='#9b59b6', alpha=0.8, edgecolor='white')
    ax.set_xticks(range(len(buckets_acc)))
    ax.set_xticklabels([f'{u:.3f}' for u in buckets_unc], rotation=45, fontsize=10)
    ax.set_xlabel('Mean Uncertainty (σ) per Bucket', fontsize=13)
    ax.set_ylabel('Accuracy (%)', fontsize=13)
    ax.set_title('Accuracy vs Uncertainty Buckets', fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    save_path = save_dir / 'uncertainty_vs_accuracy.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Uncertainty Quantification via MC Dropout")
    parser.add_argument('--model', type=str, default='hatr',
                        choices=['cnn', 'vit', 'hatr'],
                        help='Model type (default: hatr)')
    parser.add_argument('--n-passes', type=int, default=50,
                        help='Number of MC Dropout forward passes (default: 50)')
    parser.add_argument('--threshold', type=float, default=0.15,
                        help='Uncertainty threshold for flagging (default: 0.15)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 60}")
    print("UNCERTAINTY QUANTIFICATION — MC Dropout")
    print(f"{'=' * 60}")
    print(f"  Device:      {device}")
    print(f"  Model:       {args.model.upper()}")
    print(f"  MC passes:   {args.n_passes}")
    print(f"  Threshold:   {args.threshold}")

    # Load model
    checkpoint_path = CHECKPOINT_DIR / f"best_{args.model}.pth"
    if not checkpoint_path.exists():
        print(f"\n  ERROR: No checkpoint found at {checkpoint_path}")
        sys.exit(1)

    model = build_model(args.model, num_classes=2, pretrained=False).to(device)

    # Dummy forward to init dynamic params (pos_embed)
    dummy = torch.randn(1, 3, 224, 224).to(device)
    model(dummy)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Loaded checkpoint from epoch {checkpoint['epoch']}")

    # DataLoader
    num_workers = 0 if os.name == 'nt' else 2
    _, _, test_loader = get_dataloaders(batch_size=32, num_workers=num_workers)

    # Run MC Dropout
    print(f"\n  Running {args.n_passes} stochastic passes per sample...\n")
    results = classify_with_uncertainty(model, test_loader, device, n_passes=args.n_passes)

    # ------- Summary statistics -------
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    accuracy = correct / total * 100
    mean_unc = np.mean([r['uncertainty'] for r in results])
    flagged = [r for r in results if r['uncertainty'] >= args.threshold]

    correct_unc = np.mean([r['uncertainty'] for r in results if r['correct']])
    wrong_unc = np.mean([r['uncertainty'] for r in results if not r['correct']]) if any(not r['correct'] for r in results) else 0.0

    print(f"\n{'─' * 50}")
    print("  RESULTS SUMMARY")
    print(f"{'─' * 50}")
    print(f"  Total samples:          {total}")
    print(f"  MC Dropout Accuracy:    {accuracy:.2f}%")
    print(f"  Mean uncertainty:       {mean_unc:.4f}")
    print(f"  Correct predictions σ:  {correct_unc:.4f}")
    print(f"  Wrong predictions σ:    {wrong_unc:.4f}")
    print(f"  Flagged for review:     {len(flagged)} / {total} "
          f"({len(flagged)/total*100:.1f}%) [threshold={args.threshold}]")

    if flagged:
        print(f"\n  ⚠️  Flagged samples (uncertainty ≥ {args.threshold}):")
        for i, r in enumerate(flagged[:10]):
            status = '✓' if r['correct'] else '✗'
            print(f"    [{status}] P(Pneumonia)={r['mean_prob']:.3f}  "
                  f"σ={r['uncertainty']:.4f}  "
                  f"True={CLASS_NAMES[r['label']]}  Pred={CLASS_NAMES[r['pred']]}")
        if len(flagged) > 10:
            print(f"    ... and {len(flagged) - 10} more")

    # ------- Plots -------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("\n  Generating plots...")
    plot_uncertainty_distribution(results, RESULTS_DIR)
    plot_reliability_diagram(results, RESULTS_DIR)
    plot_uncertainty_vs_accuracy(results, RESULTS_DIR)

    print(f"\n{'=' * 60}")
    print("Uncertainty analysis complete! Results saved to .tmp/results/")


if __name__ == "__main__":
    main()
