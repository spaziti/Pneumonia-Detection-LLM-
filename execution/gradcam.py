"""
Phase 4b: Grad-CAM visualization for Pneumonia Detection models.

Generates Grad-CAM heatmap overlays on chest X-ray images to visualize
which regions the model focuses on for its predictions.
Supports interpretability analysis for clinical trust.

Usage:
    python gradcam.py                     # Grad-CAM for HATR model
    python gradcam.py --model cnn         # Grad-CAM for CNN baseline
    python gradcam.py --num-samples 8     # Number of sample images
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess import (
    CLASS_NAMES, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD
)
from model import build_model

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / ".tmp" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / ".tmp" / "results"
DATA_DIR = PROJECT_ROOT / ".tmp" / "data" / "chest_xray"


class GradCAM:
    """
    Grad-CAM implementation for CNN and hybrid CNN-ViT models.
    Computes class activation maps by using gradients of the target class
    flowing into the final convolutional layer.

    Supports context manager usage to automatically clean up hooks:
        with GradCAM(model, target_layer) as gc:
            heatmap, cls, conf = gc.generate(tensor)
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks and STORE handles for cleanup
        self._fwd_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        """Remove registered hooks to prevent memory leaks."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()
        return False

    def generate(self, input_tensor, target_class=None, tabular=None):
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: (1, 3, H, W) input image tensor
            target_class: Target class index (None = predicted class)
            tabular: Optional clinical tabular features tensor

        Returns:
            heatmap: (H, W) normalized heatmap
            predicted_class: Predicted class index
            confidence: Prediction confidence
        """
        self.model.eval()

        # Forward pass
        if tabular is not None:
            output = self.model(input_tensor, tabular)
        else:
            output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        confidence = probs[0, target_class].item()

        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()

        # Compute Grad-CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU and normalize
        cam = F.relu(cam)
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input size
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        return cam, target_class, confidence


def get_target_layer(model, model_type):
    """Get the target convolutional layer for Grad-CAM."""
    if model_type == 'cnn':
        # Last conv layer of ResNet backbone
        last_block = model.backbone.layer4[-1]
        if hasattr(last_block, 'conv3'):
            return last_block.conv3
        return last_block.conv2
    elif model_type == 'hatr':
        # Last conv layer of the CNN backbone
        last_block = model.cnn_backbone[-1][-1]
        if hasattr(last_block, 'conv3'):
            return last_block.conv3
        return last_block.conv2
    elif model_type == 'vit':
        # For ViT, use the patch embedding layer (limited but functional)
        return model.vit.patch_embed.proj
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_and_preprocess_image(img_path):
    """Load an image and return both raw and preprocessed versions."""
    # Raw image for display
    raw_img = Image.open(img_path).convert("RGB")
    raw_img = raw_img.resize((IMG_SIZE, IMG_SIZE))
    raw_np = np.array(raw_img)

    # Preprocessed tensor for model
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tensor = transform(raw_img).unsqueeze(0)

    return raw_np, tensor


def create_gradcam_visualization(model, model_type, device, num_samples=8):
    """Generate Grad-CAM visualizations for sample test images."""

    # Collect sample images from test set
    samples = []
    for class_name in CLASS_NAMES:
        class_dir = DATA_DIR / "test" / class_name
        if class_dir.exists():
            images = sorted(class_dir.iterdir())[:num_samples // 2]
            for img_path in images:
                if img_path.suffix.lower() in ['.jpeg', '.jpg', '.png']:
                    samples.append((str(img_path), class_name))

    if not samples:
        print("    No test images found!")
        return

    # Limit to num_samples
    samples = samples[:num_samples]

    # Generate visualizations
    n_cols = 4
    n_rows = (len(samples) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'Grad-CAM Visualization — {model_type.upper()} Model',
                 fontsize=18, fontweight='bold', y=1.02)

    # Check if model is multimodal
    is_multimodal = hasattr(model, 'tabular_encoder')
    ehr_records = {}
    if is_multimodal:
        import json
        ehr_path = PROJECT_ROOT / ".tmp" / "data" / "ehr_records.json"
        if ehr_path.exists():
            with open(ehr_path, 'r') as f:
                ehr_records = json.load(f)

    # Use context manager to auto-cleanup hooks after all images are processed
    target_layer = get_target_layer(model, model_type)
    with GradCAM(model, target_layer) as grad_cam:
        for idx, (img_path, true_label) in enumerate(samples):
            row, col = idx // n_cols, idx % n_cols

            # Load image
            raw_img, tensor = load_and_preprocess_image(img_path)
            tensor = tensor.to(device)

            # Generate Grad-CAM
            tabular_tensor = None
            if is_multimodal:
                filename = Path(img_path).name
                record = ehr_records.get(filename, {})
                EHR_FIELDS = [
                    'age', 'temperature', 'heart_rate', 'wbc_count',
                    'respiratory_rate', 'cough_duration_days', 'oxygen_saturation'
                ]
                tab_vals = [float(record.get(field, 0.0)) for field in EHR_FIELDS]
                tabular_tensor = torch.FloatTensor([tab_vals]).to(device)

            heatmap, pred_class, confidence = grad_cam.generate(tensor, tabular=tabular_tensor)

            # Overlay heatmap on original image
            ax = axes[row, col]

            ax.imshow(raw_img)
            ax.imshow(heatmap, alpha=0.4, cmap='jet')

            pred_label = CLASS_NAMES[pred_class]
            correct = pred_label == true_label
            color = '#2ecc71' if correct else '#e74c3c'
            symbol = '✓' if correct else '✗'

            ax.set_title(
                f"True: {true_label}\n"
                f"Pred: {pred_label} ({confidence:.1%}) {symbol}",
                fontsize=10, color=color, fontweight='bold'
            )
            ax.axis('off')

    # Hide unused axes
    for idx in range(len(samples), n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')

    save_path = RESULTS_DIR / f'gradcam_{model_type}.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM Visualization")
    parser.add_argument('--model', type=str, default='hatr',
                        choices=['cnn', 'vit', 'hatr', 'all'],
                        help='Model type (default: hatr)')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet50'],
                        help='CNN backbone type (default: resnet18)')
    parser.add_argument('--num-samples', type=int, default=8,
                        help='Number of sample images (default: 8)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 60}")
    print(f"PHASE 4b: Grad-CAM Visualization")
    print(f"{'=' * 60}")
    print(f"  Device: {device}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    models_to_viz = ['cnn', 'hatr'] if args.model == 'all' else [args.model]

    for model_type in models_to_viz:
        # First try loading specific checkpoint
        checkpoint_path = CHECKPOINT_DIR / f"best_{model_type}_{args.backbone}.pth"
        if not checkpoint_path.exists():
            # Fallback to default
            checkpoint_path = CHECKPOINT_DIR / f"best_{model_type}.pth"

        if not checkpoint_path.exists():
            print(f"\n  No checkpoint found for {model_type} at {checkpoint_path}")
            continue

        print(f"\n  Generating Grad-CAM for {model_type.upper()}...")

        # Load checkpoint and determine the backbone_type from metadata
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        actual_backbone = checkpoint.get('backbone_type', args.backbone)

        # Auto-detect if checkpoint is multimodal
        state_keys = checkpoint.get("model_state_dict", {}).keys()
        is_multimodal = any(k.startswith("tabular_encoder") for k in state_keys)

        # Load model (pos_embed is now initialized in __init__, no dummy forward needed)
        model = build_model(
            model_type, num_classes=2, pretrained=False,
            multimodal=is_multimodal, backbone_type=actual_backbone
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        create_gradcam_visualization(model, model_type, device, args.num_samples)

    print(f"\n{'=' * 60}")
    print("Grad-CAM visualization complete! Results saved to .tmp/results/")


if __name__ == "__main__":
    main()
