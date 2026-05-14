"""
Phase 2: Model architectures for Pneumonia Detection.

Three model variants as specified in the URP Report:
1. CNN-Only:    ResNet-18 backbone (baseline ~87.3%)
2. ViT-Only:    Vision Transformer Small from timm
3. HATR-Hybrid: CNN backbone → Overlapping tokenization → ViT encoder
                → Adaptive fusion → Classifier (target ~91.4%)

The HATR (Hierarchical Adaptive Token Refinement) fusion strategy:
- Converts CNN feature maps to transformer tokens via overlapping sliding windows
- Uses lightweight attention weighting to suppress irrelevant background
- Hierarchical token refinement before final classification

Usage:
    from model import build_model
    model = build_model("hatr", num_classes=2, pretrained=True)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
import math


class CNNOnlyModel(nn.Module):
    """
    Baseline CNN model using ResNet-18.
    Expected accuracy: ~87.3% per the reference paper.
    """

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        # Load pretrained ResNet-18
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features  # 512

        # Replace classification head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def get_feature_extractor(self):
        """Return backbone without FC for Grad-CAM."""
        layers = list(self.backbone.children())[:-1]
        return nn.Sequential(*layers)


class ViTOnlyModel(nn.Module):
    """
    Vision Transformer model using timm's ViT-Small.
    Adapted for medical imaging with smaller patch size.
    """

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.vit = timm.create_model(
            'vit_small_patch16_224',
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=0.2,
            attn_drop_rate=0.1
        )

    def forward(self, x):
        return self.vit(x)


class OverlappingTokenizer(nn.Module):
    """
    Overlapping sliding window tokenizer for CNN-to-ViT conversion.
    Preserves spatial continuity by using overlapping windows,
    preventing fragmentation of continuous anatomical structures.
    """

    def __init__(self, in_channels, token_dim, window_size=3, stride=1, padding=1):
        super().__init__()
        self.tokenizer = nn.Conv2d(
            in_channels, token_dim,
            kernel_size=window_size,
            stride=stride,
            padding=padding
        )
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, feature_map):
        """
        Args:
            feature_map: (B, C, H, W) CNN feature maps
        Returns:
            tokens: (B, N, D) where N = H*W, D = token_dim
        """
        tokens = self.tokenizer(feature_map)  # (B, D, H, W)
        B, D, H, W = tokens.shape
        tokens = tokens.flatten(2).transpose(1, 2)  # (B, N, D)
        tokens = self.norm(tokens)
        return tokens


class AdaptiveTokenAttention(nn.Module):
    """
    Lightweight attention module for adaptive token weighting.
    Suppresses irrelevant background signals and highlights
    diagnostically relevant regions.
    """

    def __init__(self, token_dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.norm = nn.LayerNorm(token_dim)
        self.gate = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.Sigmoid()
        )

    def forward(self, tokens):
        """Apply attention-based token refinement."""
        # Self-attention
        attended, _ = self.attention(tokens, tokens, tokens)
        tokens = self.norm(tokens + attended)

        # Gated refinement - suppress irrelevant tokens
        gate_weights = self.gate(tokens)
        tokens = tokens * gate_weights

        return tokens


class HierarchicalTokenRefinement(nn.Module):
    """
    Hierarchical Adaptive Token Refinement (HATR) module.
    Multi-level refinement of tokens before classification.
    """

    def __init__(self, token_dim, num_levels=2, num_heads=4):
        super().__init__()
        self.levels = nn.ModuleList([
            AdaptiveTokenAttention(token_dim, num_heads)
            for _ in range(num_levels)
        ])
        self.level_norms = nn.ModuleList([
            nn.LayerNorm(token_dim)
            for _ in range(num_levels)
        ])

    def forward(self, tokens):
        """Apply hierarchical multi-level refinement."""
        for level, norm in zip(self.levels, self.level_norms):
            refined = level(tokens)
            tokens = norm(tokens + refined)  # Residual connection
        return tokens


class TransformerEncoder(nn.Module):
    """Lightweight transformer encoder for token processing."""

    def __init__(self, token_dim, num_layers=4, num_heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=int(token_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)
        self.pos_embed = None  # Will be initialized in forward

    def forward(self, tokens):
        B, N, D = tokens.shape

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (B, N+1, D)

        # Add positional embedding
        if self.pos_embed is None or self.pos_embed.shape[1] != N + 1:
            self.pos_embed = nn.Parameter(
                torch.randn(1, N + 1, D, device=tokens.device) * 0.02
            )
        tokens = tokens + self.pos_embed

        # Encode
        tokens = self.encoder(tokens)

        # Return CLS token representation
        return tokens[:, 0]  # (B, D)


class HATRHybridModel(nn.Module):
    """
    Hybrid CNN-ViT model with Hierarchical Adaptive Token Refinement (HATR).

    Architecture flow:
    Input (224x224x3)
      → CNN Backbone (ResNet-18, features only) → Feature maps (512, 7, 7)
      → Overlapping Tokenizer → Tokens (49, 384)
      → HATR Refinement → Refined tokens (49, 384)
      → Transformer Encoder → CLS representation (384)
      → Classification Head → Logits (2)

    Target accuracy: ~91.4% per the reference paper.
    """

    def __init__(self, num_classes=2, pretrained=True, token_dim=384):
        super().__init__()

        # CNN Backbone: ResNet-18 feature extractor (without FC and avgpool)
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])
        cnn_out_channels = 512  # ResNet-18 final layer channels

        # Overlapping tokenizer: CNN features → transformer tokens
        self.tokenizer = OverlappingTokenizer(
            in_channels=cnn_out_channels,
            token_dim=token_dim,
            window_size=3,
            stride=1,
            padding=1
        )

        # Hierarchical Adaptive Token Refinement
        self.hatr = HierarchicalTokenRefinement(
            token_dim=token_dim,
            num_levels=2,
            num_heads=4
        )

        # Transformer encoder
        self.transformer = TransformerEncoder(
            token_dim=token_dim,
            num_layers=4,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.1
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Dropout(0.3),
            nn.Linear(token_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        # Store the backbone for Grad-CAM access
        self._cnn_backbone = self.cnn_backbone

    def forward(self, x):
        # Step 1: Extract CNN features
        features = self.cnn_backbone(x)  # (B, 512, 7, 7)

        # Step 2: Convert to tokens via overlapping windows
        tokens = self.tokenizer(features)  # (B, 49, 384)

        # Step 3: Hierarchical adaptive token refinement
        tokens = self.hatr(tokens)  # (B, 49, 384)

        # Step 4: Transformer encoding
        cls_repr = self.transformer(tokens)  # (B, 384)

        # Step 5: Classification
        logits = self.classifier(cls_repr)  # (B, 2)

        return logits

    def get_cnn_features(self, x):
        """Get CNN feature maps for Grad-CAM visualization."""
        return self.cnn_backbone(x)


def build_model(model_type="hatr", num_classes=2, pretrained=True):
    """
    Factory function to build the specified model type.

    Args:
        model_type: One of 'cnn', 'vit', 'hatr'
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights

    Returns:
        model: PyTorch model
    """
    model_map = {
        "cnn": CNNOnlyModel,
        "vit": ViTOnlyModel,
        "hatr": HATRHybridModel,
    }

    if model_type not in model_map:
        raise ValueError(f"Unknown model type '{model_type}'. Choose from: {list(model_map.keys())}")

    model = model_map[model_type](num_classes=num_classes, pretrained=pretrained)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model: {model_type.upper()}")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    return model


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 2: Model Architecture Validation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    dummy_input = torch.randn(2, 3, 224, 224).to(device)

    for model_type in ["cnn", "vit", "hatr"]:
        print(f"\n{'─' * 40}")
        model = build_model(model_type, num_classes=2, pretrained=False).to(device)
        output = model(dummy_input)
        print(f"  Input:  {dummy_input.shape}")
        print(f"  Output: {output.shape}")
        assert output.shape == (2, 2), f"Expected (2,2), got {output.shape}"
        print(f"  ✓ Forward pass successful")

    print(f"\n{'=' * 60}")
    print("All models validated successfully!")
