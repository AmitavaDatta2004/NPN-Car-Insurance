"""MobileNetV2-based fraud-risk classifier for ClaimVision AI.

Architecture (per README section 10 Notebook 02):
    224x224 RGB
    -> ImageNet-pretrained MobileNetV2 backbone
    -> Global Average Pooling  [1280-dim]
    -> Dropout(p)
    -> Linear(1280 -> 128) -> ReLU -> Dropout(p)
    -> Linear(128 -> 1)   [raw logit for BCEWithLogitsLoss]

Experiment ID: FRAUD-MNV2-001
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


class FraudClassifier(nn.Module):
    """MobileNetV2 backbone with a custom fraud-detection head.

    The backbone can be fully frozen (Stage A) or partially unfrozen
    (Stage B: features[17:] unlocked for fine-tuning).

    Args:
        pretrained: Load ImageNet weights for the backbone.
        dropout: Dropout probability applied before and after the hidden layer.
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.3) -> None:
        super().__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)

        # Feature extractor - keep all InvertedResidual blocks
        self.features = backbone.features  # output channels: 1280

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1),
        )

        self._dropout = dropout
        self._pretrained = pretrained

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, 224, 224).

        Returns:
            Raw logit tensor of shape (B, 1).
        """
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (Stage A training)."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_final_blocks(self, n_blocks: int = 2) -> None:
        """Unfreeze the last n_blocks InvertedResidual blocks (Stage B fine-tuning).

        MobileNetV2 has 19 feature layers (index 0-18).
        n_blocks=2 unfreezes features[17] and features[18].

        Args:
            n_blocks: Number of terminal blocks to unfreeze.
        """
        total = len(self.features)
        for i, block in enumerate(self.features):
            if i >= total - n_blocks:
                for param in block.parameters():
                    param.requires_grad = True

    def trainable_parameter_count(self) -> dict[str, int]:
        """Return count of trainable and total parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {"trainable": trainable, "total": total, "frozen": total - trainable}


def build_fraud_model(pretrained: bool = True, dropout: float = 0.3) -> FraudClassifier:
    """Build and return a FraudClassifier.

    Args:
        pretrained: If True, loads ImageNet weights (requires one-time internet access).
        dropout: Dropout probability.

    Returns:
        FraudClassifier with backbone frozen by default (Stage A ready).
    """
    model = FraudClassifier(pretrained=pretrained, dropout=dropout)
    model.freeze_backbone()
    return model


def load_fraud_model(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> FraudClassifier:
    """Load a FraudClassifier from a saved .pt checkpoint.

    The checkpoint must have been saved with:
        torch.save({"model_state_dict": model.state_dict(), ...}, path)

    Args:
        checkpoint_path: Path to the .pt file.
        device: Target device ("cpu" or "cuda").

    Returns:
        FraudClassifier in eval mode on the target device.

    Raises:
        FileNotFoundError: If checkpoint_path does not exist.
        KeyError: If the checkpoint lacks the expected keys.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Fraud model checkpoint not found: {path}. "
            "Run notebook 02_fraud_mobilenetv2_training.ipynb to generate it."
        )

    checkpoint = torch.load(path, map_location=device, weights_only=True)

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint at {path} is missing 'model_state_dict'. "
            "Ensure it was saved with the standard ClaimVision checkpoint format."
        )

    dropout = checkpoint.get("dropout", 0.3)
    model = FraudClassifier(pretrained=False, dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def export_onnx(
    model: FraudClassifier,
    output_path: str | Path,
    device: str = "cpu",
) -> None:
    """Export a FraudClassifier to ONNX format.

    The exported model accepts a float32 tensor of shape (B, 3, 224, 224)
    and returns a float32 logit tensor of shape (B, 1).

    Args:
        model: FraudClassifier in eval mode.
        output_path: Destination path for the .onnx file.
        device: Device the model and dummy input should be placed on.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy_input = torch.zeros(1, 3, 224, 224, device=device)

    # Use TorchScript-based exporter (dynamo=False) for cross-platform stability,
    # dynamic_axes compatibility, and avoiding Windows console charmap encoding issues.
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["image"],
            output_names=["logit"],
            dynamic_axes={"image": {0: "batch_size"}, "logit": {0: "batch_size"}},
            dynamo=False,
        )
    except TypeError:
        # Fallback for PyTorch versions where dynamo argument does not exist
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["image"],
            output_names=["logit"],
            dynamic_axes={"image": {0: "batch_size"}, "logit": {0: "batch_size"}},
        )


def save_preprocessing_config(output_path: str | Path) -> None:
    """Write the preprocessing configuration used at training time.

    Args:
        output_path: Path to write the JSON file.
    """
    config = {
        "image_size": [224, 224],
        "color_order": "RGB",
        "normalisation": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "resize_strategy": "Resize(256) + CenterCrop(224) for val/test",
        "training_augmentations": [
            "RandomResizedCrop(224, scale=(0.7, 1.0))",
            "RandomHorizontalFlip(p=0.5)",
            "ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)",
            "RandomRotation(degrees=15)",
        ],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
