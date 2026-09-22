"""location/mobilenet.py — MobileNetV2 transfer-learning location classifier.

Task ID  : LOC-MNV2-001
Phase    : 11a
Owner    : Detection ML member / Antigravity

Architecture
------------
224 × 224 RGB image
  → ImageNet-pretrained MobileNetV2 backbone (torchvision)
  → global average pooling (built into MobileNetV2)
  → dropout (0.3)
  → linear 128
  → ReLU
  → dropout (0.3)
  → 5 logits  (headlamp, front_bumper, hood, door, rear_bumper)

Training strategy (2-stage fine-tuning)
-----------------------------------------
Stage A: freeze backbone, train only classification head.
         LR ≈ 1e-3, 20–25 epochs, patience 10.
Stage B: unfreeze final 2 InvertedResidual blocks, fine-tune with LR ≈ 5e-5.
         10–15 epochs, patience 8.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNetV2

from claimvision_ml.location.dataset import LOCATION_CLASSES, LOCATION_IMAGE_SIZE


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LocationMobileNet(nn.Module):
    """MobileNetV2 fine-tuned for 5-class damaged-part location classification.

    Parameters
    ----------
    num_classes:
        Number of output classes (default 5).
    dropout:
        Dropout probability in the classification head.
    pretrained:
        If True, loads ImageNet weights (default True — always use pretrained
        for this small dataset).
    """

    def __init__(
        self,
        num_classes: int = 5,
        dropout: float = 0.3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self._num_classes = num_classes
        self._dropout = dropout

        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone: MobileNetV2 = models.mobilenet_v2(weights=weights)

        # Replace the standard classifier head
        in_features = backbone.classifier[1].in_features  # 1280
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns logits of shape (B, num_classes)."""
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        """Freeze all backbone feature layers for Stage A training."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_last_blocks(self, n_blocks: int = 2) -> None:
        """Unfreeze the last *n_blocks* InvertedResidual blocks for Stage B fine-tuning."""
        feature_layers = list(self.backbone.features.children())
        n_total = len(feature_layers)
        for layer in feature_layers[max(0, n_total - n_blocks):]:
            for param in layer.parameters():
                param.requires_grad = True

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def parameter_count(self) -> dict[str, int]:
        """Return trainable and total parameter counts."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {"trainable": trainable, "total": total, "frozen": total - trainable}

    def measure_cpu_latency(
        self,
        image: Any,
        num_runs: int = 25,
        image_size: int = LOCATION_IMAGE_SIZE,
    ) -> float:
        """Measure average CPU inference latency in milliseconds.

        Parameters
        ----------
        image:
            A PIL Image, numpy array (H×W×3), or a pre-processed tensor (1×3×H×W).
        num_runs:
            Number of forward passes to average.
        image_size:
            Target square size for preprocessing when image is not a tensor.

        Returns
        -------
        float: average latency in milliseconds.
        """
        from claimvision_ml.location.dataset import get_location_transforms
        from PIL import Image as PILImage
        import numpy as np

        self.eval()
        device = torch.device("cpu")
        self.to(device)

        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                tensor = image.unsqueeze(0).to(device)
            else:
                tensor = image.to(device)
        elif isinstance(image, np.ndarray):
            pil = PILImage.fromarray(image.astype("uint8"))
            tensor = get_location_transforms("val", image_size)(pil).unsqueeze(0).to(device)
        elif isinstance(image, PILImage.Image):
            tensor = get_location_transforms("val", image_size)(image).unsqueeze(0).to(device)
        elif isinstance(image, (str, Path)):
            pil = PILImage.open(image).convert("RGB")
            tensor = get_location_transforms("val", image_size)(pil).unsqueeze(0).to(device)
        else:
            tensor = torch.zeros(1, 3, image_size, image_size, device=device)

        # Warm-up
        with torch.no_grad():
            self(tensor)

        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                t0 = time.perf_counter()
                self(tensor)
                times.append((time.perf_counter() - t0) * 1000.0)
        return float(sum(times) / len(times))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_location_mobilenet(
    dropout: float = 0.3,
    num_classes: int = 5,
    pretrained: bool = True,
) -> LocationMobileNet:
    """Build and return a fresh :class:`LocationMobileNet`.

    Parameters
    ----------
    dropout:
        Dropout probability in the head.
    num_classes:
        Number of output classes.
    pretrained:
        Whether to use ImageNet weights (strongly recommended for tiny datasets).

    Returns
    -------
    :class:`LocationMobileNet` with backbone frozen (ready for Stage A).
    """
    model = LocationMobileNet(num_classes=num_classes, dropout=dropout, pretrained=pretrained)
    model.freeze_backbone()
    return model


def load_location_mobilenet(
    checkpoint_path: str | Path,
    num_classes: int = 5,
    dropout: float = 0.3,
    device: str = "cpu",
) -> LocationMobileNet:
    """Load a :class:`LocationMobileNet` from a saved checkpoint.

    Parameters
    ----------
    checkpoint_path:
        Path to a ``.pt`` checkpoint saved with ``torch.save(model.state_dict(), ...)``.
    num_classes:
        Must match the architecture used during training.
    dropout:
        Must match the architecture used during training.
    device:
        Target device (``"cpu"`` or ``"cuda"``).

    Returns
    -------
    :class:`LocationMobileNet` in eval mode.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = LocationMobileNet(num_classes=num_classes, dropout=dropout, pretrained=False)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model
