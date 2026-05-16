"""
Phase 1b: Preprocessing and DataLoader creation for Chest X-ray Pneumonia dataset.

- Resize to 224x224
- Normalize with ImageNet mean/std
- Data augmentation (train only): random flip, rotation, brightness/contrast
- Class-weighted sampling for imbalanced classes (3:1 ratio)
- PyTorch Dataset and DataLoader classes

Usage:
    from preprocess import get_dataloaders, get_class_weights
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)
    weights = get_class_weights(train_loader.dataset)
"""

import os
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / ".tmp" / "data" / "chest_xray"
RESULTS_DIR = PROJECT_ROOT / ".tmp" / "results"

# ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Image size as specified in the report
IMG_SIZE = 224

# Class mapping
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


class ChestXrayDataset(Dataset):
    """PyTorch Dataset for Chest X-ray Pneumonia classification."""

    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir: Path to the chest_xray directory
            split: One of 'train', 'val', 'test'
            transform: torchvision transforms to apply
        """
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.samples = []
        self.labels = []

        # Scan directory for images
        for class_name in CLASS_NAMES:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                print(f"Warning: {class_dir} not found")
                continue

            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in ['.jpeg', '.jpg', '.png']:
                    self.samples.append(str(img_path))
                    self.labels.append(CLASS_TO_IDX[class_name])

        print(f"  Loaded {split}: {len(self.samples)} images "
              f"({dict(Counter(self.labels))})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]

        # Load image as RGB
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(split="train"):
    """Get transforms for each split."""
    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def get_class_weights(dataset):
    """
    Compute class weights for weighted cross-entropy loss.
    Uses inverse frequency weighting with emphasis on pneumonia detection.
    """
    label_counts = Counter(dataset.labels)
    total = len(dataset.labels)
    n_classes = len(CLASS_NAMES)

    # Inverse frequency weighting
    weights = []
    for i in range(n_classes):
        count = label_counts.get(i, 1)
        w = total / (n_classes * count)
        weights.append(w)

    weights = torch.FloatTensor(weights)
    print(f"  Class weights: {dict(zip(CLASS_NAMES, weights.tolist()))}")
    return weights


def get_weighted_sampler(dataset):
    """Create a WeightedRandomSampler for balanced training."""
    label_counts = Counter(dataset.labels)
    total = len(dataset.labels)

    # Weight per sample (inverse of class frequency)
    class_weights = {cls: total / count for cls, count in label_counts.items()}
    sample_weights = [class_weights[label] for label in dataset.labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


class MultiModalChestXrayDataset(ChestXrayDataset):
    """
    Extends ChestXrayDataset to also return tabular EHR features.
    Returns (image, tabular_tensor, label) triplets.
    """

    EHR_FIELDS = [
        'age', 'temperature', 'heart_rate', 'wbc_count',
        'respiratory_rate', 'cough_duration_days', 'oxygen_saturation'
    ]

    def __init__(self, root_dir, split="train", transform=None, ehr_path=None):
        super().__init__(root_dir, split, transform)

        # Load EHR records
        if ehr_path is None:
            ehr_path = PROJECT_ROOT / ".tmp" / "data" / "ehr_records.json"

        import json
        if Path(ehr_path).exists():
            with open(ehr_path, 'r') as f:
                self.ehr_records = json.load(f)
            print(f"  Loaded {len(self.ehr_records)} EHR records from {ehr_path}")
        else:
            print(f"  WARNING: EHR file not found at {ehr_path}, using zero vectors")
            self.ehr_records = {}

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)

        # Look up EHR record by filename
        filename = Path(self.samples[idx]).name
        record = self.ehr_records.get(filename, {})

        tabular = torch.FloatTensor([
            record.get(field, 0.0) for field in self.EHR_FIELDS
        ])

        return image, tabular, label


def get_dataloaders(data_dir=None, batch_size=32, num_workers=2, multimodal=False):
    """
    Create train, val, and test DataLoaders.

    Args:
        data_dir: Path to chest_xray dataset
        batch_size: Batch size
        num_workers: Number of dataloader workers
        multimodal: If True, uses MultiModalChestXrayDataset (returns image+tabular+label)

    Returns:
        train_loader, val_loader, test_loader
    """
    if data_dir is None:
        data_dir = DATA_DIR

    print("Creating DataLoaders...")

    DatasetClass = MultiModalChestXrayDataset if multimodal else ChestXrayDataset

    # Create datasets
    train_dataset = DatasetClass(data_dir, "train", get_transforms("train"))
    val_dataset = DatasetClass(data_dir, "val", get_transforms("val"))
    test_dataset = DatasetClass(data_dir, "test", get_transforms("test"))

    if len(train_dataset) == 0:
        raise RuntimeError("No training images found! Ensure you have successfully run download_dataset.py first.")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def save_sample_images(data_dir=None):
    """Save sample images from the dataset for visual inspection."""
    if data_dir is None:
        data_dir = DATA_DIR

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Sample Chest X-ray Images", fontsize=16, fontweight='bold')

    for row, class_name in enumerate(CLASS_NAMES):
        class_dir = data_dir / "train" / class_name
        images = sorted(class_dir.iterdir())[:4]

        for col, img_path in enumerate(images):
            img = Image.open(img_path).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE))
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].set_title(f"{class_name}", fontsize=12)
            axes[row, col].axis('off')

    plt.tight_layout()
    save_path = RESULTS_DIR / "sample_images.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Sample images saved to {save_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 1b: Preprocessing & DataLoader Setup")
    print("=" * 60)

    # Create dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)

    print(f"\n  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    # Get class weights
    weights = get_class_weights(train_loader.dataset)

    # Save sample images
    print("\nSaving sample images...")
    save_sample_images()

    # Test a batch
    print("\nTesting batch loading...")
    batch_imgs, batch_labels = next(iter(train_loader))
    print(f"  Batch shape: {batch_imgs.shape}")
    print(f"  Labels: {batch_labels[:8].tolist()}")
    print("\nPreprocessing complete!")
