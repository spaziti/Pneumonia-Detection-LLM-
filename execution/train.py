"""
Phase 3: Training pipeline for Pneumonia Detection models.

Training configuration (from URP Report):
- Optimizer: Adam, lr=1e-4
- Loss: Weighted cross-entropy (class weights from dataset distribution)
- Batch size: 32
- Scheduler: CosineAnnealingLR
- Early stopping with patience

Trains all three model variants (CNN, ViT, HATR) for comparative analysis.

Usage:
    python train.py                          # Train HATR model (default)
    python train.py --model cnn              # Train CNN-only baseline
    python train.py --model vit              # Train ViT-only model
    python train.py --model all              # Train all three models
    python train.py --epochs 10 --model hatr # Quick run
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add execution directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess import get_dataloaders, get_class_weights
from model import build_model

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / ".tmp" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / ".tmp" / "results"
LOG_DIR = PROJECT_ROOT / ".tmp" / "logs"


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs, multimodal=False):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"  Train Epoch {epoch+1}/{total_epochs}",
                leave=False, ncols=100)

    for batch_idx, batch in enumerate(pbar):
        if multimodal:
            images, tabular, labels = batch
            images, tabular, labels = images.to(device), tabular.to(device), labels.to(device)
            outputs = model(images, tabular)
        else:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/(batch_idx+1):.4f}',
            'acc': f'{100.*correct/total:.1f}%'
        })

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device, multimodal=False):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        if multimodal:
            images, tabular, labels = batch
            images, tabular, labels = images.to(device), tabular.to(device), labels.to(device)
            outputs = model(images, tabular)
        else:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

        loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / len(loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc


def train_model(model_type, epochs=25, batch_size=32, lr=1e-4, patience=7,
                device=None, backbone_ckpt=None, multimodal=False):
    """
    Full training loop for a given model type.

    Args:
        model_type: 'cnn', 'vit', or 'hatr'
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        patience: Early stopping patience
        device: torch device
        backbone_ckpt: Path to SimCLR pre-trained backbone (optional)
        multimodal: If True, uses multi-modal (image + EHR) pipeline

    Returns:
        history: dict with training metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create directories
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"TRAINING: {model_type.upper()} Model")
    print(f"{'=' * 60}")
    print(f"  Device:     {device}")
    print(f"  Epochs:     {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  LR:         {lr}")
    print(f"  Patience:   {patience}")

    # DataLoaders
    num_workers = 0 if os.name == 'nt' else 2  # Windows compatibility
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size, num_workers=num_workers, multimodal=multimodal
    )

    # Model
    model = build_model(model_type, num_classes=2, pretrained=True,
                         backbone_ckpt=backbone_ckpt, multimodal=multimodal).to(device)

    # Loss with class weights
    class_weights = get_class_weights(train_loader.dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer: Adam as specified in report
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

    # Scheduler: Cosine annealing
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # Training history
    history = {
        'model_type': model_type,
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': [],
        'best_val_acc': 0.0,
        'best_epoch': 0,
        'total_time': 0.0
    }

    # Early stopping
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()

    print("\n  Starting training...\n")

    for epoch in range(epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, epochs,
            multimodal=multimodal
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device,
                                     multimodal=multimodal)

        # Step scheduler
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        epoch_time = time.time() - epoch_start

        print(f"  Epoch {epoch+1:>3}/{epochs} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:>6.2f}% | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:>6.2f}% | "
              f"LR: {current_lr:.2e} | {epoch_time:.1f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            history['best_val_acc'] = best_val_acc
            history['best_epoch'] = epoch + 1
            patience_counter = 0

            checkpoint_path = CHECKPOINT_DIR / f"best_{model_type}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'model_type': model_type,
            }, checkpoint_path)
            print(f"  * Best model saved (val_acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping triggered after {patience} epochs without improvement.")
                break

    total_time = time.time() - start_time
    history['total_time'] = total_time

    print(f"\n{'-' * 60}")
    print(f"  Training complete for {model_type.upper()}")
    print(f"  Best val accuracy: {best_val_acc:.2f}% (epoch {history['best_epoch']})")
    print(f"  Total training time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Save history
    history_path = LOG_DIR / f"history_{model_type}.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  History saved to {history_path}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train Pneumonia Detection Models")
    parser.add_argument('--model', type=str, default='hatr',
                        choices=['cnn', 'vit', 'hatr', 'all'],
                        help='Model type to train (default: hatr)')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of training epochs (default: 25)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--patience', type=int, default=7,
                        help='Early stopping patience (default: 7)')
    parser.add_argument('--backbone-ckpt', type=str, default=None,
                        help='Path to SimCLR pre-trained backbone checkpoint')
    parser.add_argument('--multimodal', action='store_true',
                        help='Enable multi-modal training with EHR data')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    models_to_train = ['cnn', 'vit', 'hatr'] if args.model == 'all' else [args.model]
    all_histories = {}

    for model_type in models_to_train:
        history = train_model(
            model_type=model_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            device=device,
            backbone_ckpt=args.backbone_ckpt,
            multimodal=args.multimodal
        )
        all_histories[model_type] = history

    # Print summary comparison
    if len(all_histories) > 1:
        print(f"\n{'=' * 60}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'=' * 60}")
        print(f"  {'Model':<10} {'Best Val Acc':>12} {'Best Epoch':>12} {'Time':>10}")
        print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*10}")
        for name, hist in all_histories.items():
            print(f"  {name.upper():<10} {hist['best_val_acc']:>11.2f}% "
                  f"{hist['best_epoch']:>12} "
                  f"{hist['total_time']:>9.1f}s")

    print("\nTraining pipeline complete!")


if __name__ == "__main__":
    main()
