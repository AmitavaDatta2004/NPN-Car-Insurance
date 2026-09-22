"""location/efficientnet.py — EfficientNet-B0 transfer-learning location classifier.

Task ID  : LOC-EFF-001
Phase    : 11b
Owner    : Detection ML member / Antigravity

Architecture
------------
224 × 224 RGB image
  → ImageNet-pretrained EfficientNet-B0 backbone (timm)
  → global average pooling (built-in)
  → dropout (0.3)
  → linear 128
  → ReLU
  → dropout (0.3)
  → 5 logits  (headlamp, front_bumper, hood, door, rear_bumper)

Training strategy (2-stage fine-tuning)
-----------------------------------------
Stage A: freeze backbone blocks 0–5, train only classifier head.
         LR ≈ 1e-3, 20–25 epochs, patience 10.
Stage B: unfreeze blocks 6–7 (last 2 of 8 total) and classifier, fine-tune.
         LR ≈ 5e-5, 10–15 epochs, patience 8.

Why EfficientNet-B0?
--------------------
- 5.3 M parameters vs MobileNetV2's 3.4 M — still compact.
- Superior compound scaling; typically higher accuracy on small datasets.
- Available via `timm` (already a dependency from ViT-Tiny severity experiments).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from claimvision_ml.location.dataset import LOCATION_CLASSES, LOCATION_IMAGE_SIZE


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LocationEfficientNet(nn.Module):
    """EfficientNet-B0 fine-tuned for 5-class damaged-part location classification.

    Parameters
    ----------
    num_classes:
        Number of output classes (default 5).
    dropout:
        Dropout probability in the classification head.
    pretrained:
        If True, loads ImageNet1k weights via timm (default True).
    """

    def __init__(
        self,
        num_classes: int = 5,
        dropout: float = 0.3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        import timm  # imported here so other modules don't need timm at import time

        self._num_classes = num_classes
        self._dropout = dropout

        # Load EfficientNet-B0 via timm (no pre-built head — we provide our own)
        self._backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,          # remove default head → output is feature vector
            global_pool="avg",      # global average pool built in
        )
        feature_dim: int = self._backbone.num_features  # 1280 for EfficientNet-B0

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns logits of shape (B, num_classes)."""
        features = self._backbone(x)   # (B, 1280) after global avg pool
        return self.classifier(features)

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters for Stage A training."""
        for param in self._backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_blocks(self, n_blocks: int = 2) -> None:
        """Unfreeze the last *n_blocks* EfficientNet blocks for Stage B fine-tuning.

        EfficientNet-B0 has 8 MBConv blocks (blocks.0 … blocks.7).  This method
        unfreezes ``blocks.{8-n_blocks}`` through ``blocks.7`` and the conv_head.
        """
        blocks = list(self._backbone.blocks.children())
        n_total = len(blocks)
        for block in blocks[max(0, n_total - n_blocks):]:
            for param in block.parameters():
                param.requires_grad = True
        # Also unfreeze conv_head and bn2 which follow the blocks
        for module in [self._backbone.conv_head, self._backbone.bn2]:
            for param in module.parameters():
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
        """Measure average CPU inference latency in milliseconds."""
        from claimvision_ml.location.dataset import get_location_transforms
        from PIL import Image as PILImage
        import numpy as np

        self.eval()
        device = torch.device("cpu")
        self.to(device)

        if isinstance(image, torch.Tensor):
            tensor = image.unsqueeze(0).to(device) if image.dim() == 3 else image.to(device)
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

def build_location_efficientnet(
    dropout: float = 0.3,
    num_classes: int = 5,
    pretrained: bool = True,
) -> LocationEfficientNet:
    """Build and return a fresh :class:`LocationEfficientNet`.

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
    :class:`LocationEfficientNet` with backbone frozen (ready for Stage A).
    """
    model = LocationEfficientNet(
        num_classes=num_classes, dropout=dropout, pretrained=pretrained
    )
    model.freeze_backbone()
    return model


def load_location_efficientnet(
    checkpoint_path: str | Path,
    num_classes: int = 5,
    dropout: float = 0.3,
    device: str = "cpu",
) -> LocationEfficientNet:
    """Load a :class:`LocationEfficientNet` from a saved checkpoint.

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
    :class:`LocationEfficientNet` in eval mode.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = LocationEfficientNet(num_classes=num_classes, dropout=dropout, pretrained=False)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model
