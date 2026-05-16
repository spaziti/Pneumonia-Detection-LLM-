"""
Feature 4: Self-Supervised Pre-training with SimCLR (Simple Contrastive Learning).

Pre-trains the ResNet-18 backbone using contrastive learning on *unlabeled*
chest X-rays.  The learned representations capture general radiographic
structure without any class labels, giving the downstream HATR model a
stronger initialisation than random or even ImageNet weights.

Usage:
    python simclr_pretrain.py --epochs 20 --batch-size 64
    python simclr_pretrain.py --epochs 50 --external-data D:/nih_chestxray/images
"""

import os
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / ".tmp" / "data" / "chest_xray"
CHECKPOINT_DIR = PROJECT_ROOT / ".tmp" / "checkpoints"


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------

class SimCLRAugmentation:
    """
    Strong stochastic augmentations for contrastive learning.
    Returns TWO different augmented views of the same image.
    """

    def __init__(self, img_size=224):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __call__(self, image):
        return self.transform(image), self.transform(image)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SimCLRDataset(Dataset):
    """
    Loads images WITHOUT labels for self-supervised pre-training.
    Returns two augmented views per image.
    """

    def __init__(self, data_dir, augmentation):
        self.augmentation = augmentation
        self.image_paths = []

        # Scan train directory (ignore labels)
        train_dir = Path(data_dir) / "train"
        if train_dir.exists():
            for img_path in sorted(train_dir.rglob("*")):
                if img_path.suffix.lower() in ['.jpeg', '.jpg', '.png']:
                    self.image_paths.append(str(img_path))

        print(f"  SimCLR dataset: {len(self.image_paths)} unlabeled images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        view1, view2 = self.augmentation(image)
        return view1, view2


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """MLP projection head: backbone_dim -> 256 -> 128."""

    def __init__(self, in_dim=512, hidden_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimCLRModel(nn.Module):
    """ResNet-18 backbone + projection head for contrastive learning."""

    def __init__(self, pretrained_imagenet=False):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained_imagenet else None
        resnet = models.resnet18(weights=weights)

        # Backbone: everything except the final FC
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # -> (B, 512, 1, 1)
        self.projection = ProjectionHead(in_dim=512, hidden_dim=256, out_dim=128)

    def forward(self, x):
        h = self.backbone(x).squeeze(-1).squeeze(-1)   # (B, 512)
        z = self.projection(h)                           # (B, 128)
        return h, F.normalize(z, dim=1)


# ---------------------------------------------------------------------------
# NT-Xent Loss (Normalised Temperature-scaled Cross-Entropy)
# ---------------------------------------------------------------------------

def nt_xent_loss(z_i, z_j, temperature=0.5):
    """
    Compute the NT-Xent contrastive loss for a batch.

    Args:
        z_i, z_j: (B, D) L2-normalised projection vectors for two views
        temperature: softmax temperature scalar

    Returns:
        loss: scalar tensor
    """
    B = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)  # (2B, D)

    # Cosine similarity matrix
    sim = torch.mm(z, z.t()) / temperature  # (2B, 2B)

    # Mask out self-similarity
    mask = torch.eye(2 * B, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)

    # Positive pairs: (i, i+B) and (i+B, i)
    pos_i = torch.arange(B, device=z.device)
    pos_j = pos_i + B
    positives = torch.cat([
        sim[pos_i, pos_j].unsqueeze(1),
        sim[pos_j, pos_i].unsqueeze(1),
    ], dim=0)  # (2B, 1)

    # All negatives
    logits = torch.cat([positives, sim[torch.arange(2*B)].masked_select(~mask[torch.arange(2*B)]).view(2*B, -1)], dim=1)

    labels = torch.zeros(2 * B, dtype=torch.long, device=z.device)
    loss = F.cross_entropy(logits, labels)
    return loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def pretrain_simclr(epochs=20, batch_size=64, lr=3e-4, temperature=0.5,
                    data_dir=None, device=None):
    """Full SimCLR pre-training loop."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if data_dir is None:
        data_dir = DATA_DIR

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("SIMCLR SELF-SUPERVISED PRE-TRAINING")
    print(f"{'=' * 60}")
    print(f"  Device:      {device}")
    print(f"  Epochs:      {epochs}")
    print(f"  Batch size:  {batch_size}")
    print(f"  LR:          {lr}")
    print(f"  Temperature: {temperature}")

    # Dataset & loader
    augmentation = SimCLRAugmentation(img_size=224)
    dataset = SimCLRDataset(data_dir, augmentation)

    num_workers = 0 if os.name == 'nt' else 2
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )

    # Model
    model = SimCLRModel(pretrained_imagenet=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters:  {total_params:,}")
    print(f"  Batches/epoch: {len(loader)}")

    # Training
    start_time = time.time()
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(loader, desc=f"  Epoch {epoch+1}/{epochs}", leave=False, ncols=100)
        for view1, view2 in pbar:
            view1, view2 = view1.to(device), view2.to(device)

            _, z1 = model(view1)
            _, z2 = model(view2)

            loss = nt_xent_loss(z1, z2, temperature)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        lr_now = optimizer.param_groups[0]['lr']

        print(f"  Epoch {epoch+1:>3}/{epochs} | Loss: {avg_loss:.4f} | LR: {lr_now:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save backbone weights only (no projection head)
            backbone_state = {}
            for k, v in model.backbone.state_dict().items():
                backbone_state[k] = v

            save_path = CHECKPOINT_DIR / "simclr_backbone.pth"
            torch.save({
                'epoch': epoch + 1,
                'backbone_state_dict': backbone_state,
                'loss': best_loss,
            }, save_path)
            print(f"  * Best backbone saved (loss: {best_loss:.4f})")

    total_time = time.time() - start_time
    print(f"\n{'─' * 60}")
    print("  SimCLR pre-training complete!")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Backbone saved to: {CHECKPOINT_DIR / 'simclr_backbone.pth'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SimCLR Self-Supervised Pre-training")
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of pre-training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='NT-Xent temperature (default: 0.5)')
    parser.add_argument('--external-data', type=str, default=None,
                        help='Path to external unlabeled dataset (e.g. NIH ChestX-ray14)')
    args = parser.parse_args()

    data_dir = args.external_data if args.external_data else DATA_DIR
    pretrain_simclr(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        data_dir=data_dir,
    )


if __name__ == "__main__":
    main()
