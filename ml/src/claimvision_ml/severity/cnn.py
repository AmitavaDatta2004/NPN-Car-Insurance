"""Severity Baseline CNN for ClaimVision AI.

Experiment ID: SEV-CNN-001
Phase: 5  (notebook 06_severity_cnn_training.ipynb)

Architecture (per README §10 Notebook 06):
    160×160 RGB
      → Conv2d(3→32)  + BatchNorm2d + ReLU + MaxPool2d(2,2)
      → Conv2d(32→64) + BatchNorm2d + ReLU + MaxPool2d(2,2)
      → Conv2d(64→128)+ BatchNorm2d + ReLU + MaxPool2d(2,2)
      → Conv2d(128→256)+ BatchNorm2d + ReLU   [optional 4th block]
      → AdaptiveAvgPool2d(1,1)                 [global avg pool]
      → Flatten → Linear(256→128) + ReLU + Dropout
      → Linear(128→3)                          [three severity logits]

Scientific note:
    ImageNet normalisation is applied for numeric stability even though
    this is a from-scratch model. No ImageNet knowledge is transferred.
    The choice is clearly labelled in the preprocessing config.

Outputs from this module are used by:
    - notebooks/06_severity_cnn_training.ipynb  (judge presentation)
    - notebooks/09_severity_model_comparison.ipynb (model selection)
    - ml/src/claimvision_ml/pipeline/  (Phase 11 unified inference)
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SEVERITY_CLASSES: list[str] = ["minor", "moderate", "severe"]
SEVERITY_CLASS_TO_ID: dict[str, int] = {"minor": 0, "moderate": 1, "severe": 2}
SEVERITY_ID_TO_CLASS: dict[int, str] = {0: "minor", 1: "moderate", 2: "severe"}

# ImageNet statistics used for normalization (numeric stability, not knowledge transfer)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# Default image size for legacy square baseline
CNN_IMAGE_SIZE: int = 160
# new improvement: native 4:3 aspect ratio matching dataset median (194x259)
CNN_IMAGE_SIZE_4_3: tuple[int, int] = (192, 256)


# ─────────────────────────────────────────────────────────────────────────────
# Path resolution — cross-platform (Windows / Colab / Linux)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_severity_image_path(raw_path: str) -> Path:
    """Resolve a severity image path across Windows, Linux, and Colab.

    Manifest ``image_path`` values are repo-relative POSIX strings such as:
        ``data/raw/car_damage_severity/data3a/training/01-minor/0001.JPEG``

    This function tries them in several plausible base directories so the
    same manifest CSV works on every platform without modification.

    Args:
        raw_path: Raw path string from the manifest CSV.

    Returns:
        A :class:`Path` pointing to the file.  If not found, returns the
        original path as a ``Path`` object (caller decides how to handle).
    """
    p = Path(raw_path)
    if p.is_file():
        return p

    # Normalise separators
    norm = str(raw_path).replace("\\", "/")

    # Strategy 1: try relative to known base directories
    marker = "data/raw/"
    if marker in norm:
        rel = norm[norm.index(marker):]
        candidates = [
            Path.cwd(),
            Path("/content/NPN-Car-Insurance"),
            Path.cwd().parent,
            Path.cwd().parent.parent,
        ]
        for base in candidates:
            candidate = (base / rel).resolve()
            if candidate.is_file():
                return candidate

    # Strategy 2: filename-only fallback inside known severity dataset dirs
    fname = Path(norm).name
    search_dirs = [
        Path("data/raw/car_damage_severity"),
        Path("/content/NPN-Car-Insurance/data/raw/car_damage_severity"),
    ]
    for d in search_dirs:
        for found in d.rglob(fname):
            if found.is_file():
                return found

    return p  # not found — let caller warn or skip


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SeverityDataset(Dataset):
    """PyTorch Dataset for the severity manifests produced by Phase 4.

    Reads a manifest CSV (``severity_train.csv``, ``severity_val.csv``, or
    ``severity_test.csv``) and returns ``(image_tensor, label_id_tensor)``
    pairs.

    Required CSV columns:
        ``image_path``  – repo-relative POSIX path to the image file
        ``label_id``    – integer class index (0=minor, 1=moderate, 2=severe)

    Args:
        manifest_path: Path to the manifest CSV.
        split: ``"train"``, ``"val"``, or ``"test"``.  Controls which
            transforms are applied.
        transform: Optional override transform.  If ``None``, uses the
            default transform for the given split.
        image_size: Target image size (default 160 or (192, 256)).
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: Literal["train", "val", "test"] = "train",
        transform: transforms.Compose | None = None,
        image_size: tuple[int, int] | int = CNN_IMAGE_SIZE_4_3,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        df = pd.read_csv(self.manifest_path)
        if "image_path" not in df.columns or "label_id" not in df.columns:
            raise ValueError(
                f"Manifest {manifest_path} must have 'image_path' and 'label_id' columns."
            )

        # Cross-platform resolution
        df["_resolved_path"] = df["image_path"].apply(
            lambda p: str(_resolve_severity_image_path(str(p)))
        )
        df["_exists"] = df["_resolved_path"].apply(lambda p: Path(p).is_file())

        n_missing = (~df["_exists"]).sum()
        if n_missing > 0:
            warnings.warn(
                f"{n_missing} of {len(df)} images could not be found on disk. "
                "Ensure data/raw/car_damage_severity is populated.",
                UserWarning,
                stacklevel=2,
            )

        self.df = df[df["_exists"]].reset_index(drop=True)
        self.split = split
        self.transform = (
            transform
            if transform is not None
            else get_severity_transforms(split, image_size=image_size)
        )
        self._label_counts: dict[int, int] = (
            self.df["label_id"].value_counts().to_dict()
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        image_path = Path(row["_resolved_path"])

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to open image {image_path}: {exc}"
            ) from exc

        image_tensor = self.transform(image)
        label_tensor = torch.tensor(int(row["label_id"]), dtype=torch.long)
        return image_tensor, label_tensor

    @property
    def label_counts(self) -> dict[int, int]:
        """Return ``{label_int: count}`` for the loaded manifest."""
        return self._label_counts

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for ``CrossEntropyLoss(weight=...)``.

        Returns:
            Float tensor of shape ``(3,)`` — one weight per severity class.
        """
        total = sum(self._label_counts.values())
        weights = []
        for cls_id in range(len(SEVERITY_CLASSES)):
            count = self._label_counts.get(cls_id, 1)
            weights.append(total / (len(SEVERITY_CLASSES) * count))
        return torch.tensor(weights, dtype=torch.float32)

    def __repr__(self) -> str:
        counts = self._label_counts
        return (
            f"SeverityDataset(split={self.split!r}, n={len(self)}, "
            f"minor={counts.get(0, 0)}, "
            f"moderate={counts.get(1, 0)}, "
            f"severe={counts.get(2, 0)})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────

def get_severity_transforms(
    split: Literal["train", "val", "test"],
    image_size: tuple[int, int] | int = CNN_IMAGE_SIZE_4_3,
) -> transforms.Compose:
    """Return the augmentation pipeline for a given data split.

    # new improvement: full-canvas native 4:3 aspect ratio (192x256) matching dataset median (194x259)
    # new improvement: removed destructive CenterCrop so bumper, headlight, and corner damage is 100% preserved
    # new improvement: streamlined edge-preserving augmentation (no rotation or multi-scale crop blur)

    ImageNet mean/std normalisation is applied in all splits.

    Args:
        split: ``"train"``, ``"val"``, or ``"test"``.
        image_size: Target spatial size (default (192, 256) for native 4:3).

    Returns:
        ``transforms.Compose`` pipeline.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"Invalid split {split!r}. Must be 'train', 'val', or 'test'.")

    # new improvement: support both rectangular (H, W) and square resolutions
    target_size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
    normalise = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

    if split == "train":
        return transforms.Compose([
            # new improvement: full-canvas resize to native 4:3 without cropping
            transforms.Resize(target_size),
            # new improvement: clean horizontal flip (damage symmetry, zero edge blur)
            transforms.RandomHorizontalFlip(p=0.5),
            # new improvement: subtle ambient lighting jitter (no edge distortion)
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            normalise,
        ])
    else:  # val or test — deterministic full-canvas evaluation
        return transforms.Compose([
            # new improvement: no CenterCrop — retains full car width and all bumper/fender damage
            transforms.Resize(target_size),
            transforms.ToTensor(),
            normalise,
        ])


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block.

    Recalibrates channel feature maps dynamically to emphasize damage deformation
    cues over background paint and scenery textures.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.fc(x).view(b, c, 1, 1)
        return x * w


class SeverityCNN(nn.Module):
    """Baseline severity classifier trained from scratch.

    4-block convolutional network (conv → bn → relu → pool) with a 2-layer
    non-linear classification head (Linear → ReLU → Dropout → Linear).
    Also supports optional Squeeze-and-Excitation (SE) channel attention and
    direct head configurations for ablation and checkpoint compatibility.

    # new improvement: Block 4 uses 128->128 channels (extra_conv_channels=128 default),
    # eliminating the 295,000 parameter bottleneck (over 70% of network) to prevent overfitting on 1,140 images.
    # new improvement: full-canvas native 4:3 input (192x256) matching dataset median (194x259).

    Args:
        num_classes: Number of output classes (default 3).
        dropout: Dropout probability in the classifier head (default 0.4).
        use_extra_conv: If True, adds a 4th conv block (default True).
        extra_conv_channels: Channel width for block 4 (default 128 to prevent bottleneck).
        use_se: If True, attaches an SE attention block to block 4 (default False).
        direct_head: If True, uses GAP → Dropout → Linear(feature_dim, num_classes).
            If False (default), retains the 2-layer non-linear head needed to isolate
            intermediate (moderate) severity boundaries.
    """

    def __init__(
        self,
        num_classes: int = 3,
        dropout: float = 0.4,
        use_extra_conv: bool = True,
        # new improvement: extra_conv_channels defaults to 128 instead of 256, eliminating the 295k parameter bottleneck
        extra_conv_channels: int = 128,
        use_se: bool = False,
        direct_head: bool = False,
    ) -> None:
        super().__init__()
        self._num_classes = num_classes
        self._dropout = dropout
        self._use_extra_conv = use_extra_conv
        self._extra_conv_channels = extra_conv_channels
        self._use_se = use_se
        self._direct_head = direct_head

        # Block 1 — 3 → 32
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Block 2 — 32 → 64
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Block 3 — 64 → 128
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Block 4 (optional)
        # new improvement: 128 -> extra_conv_channels (default 128) avoids massive parameter explosion
        if use_extra_conv:
            layers: list[nn.Module] = [
                nn.Conv2d(128, extra_conv_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(extra_conv_channels),
                nn.ReLU(inplace=True),
            ]
            if use_se:
                layers.append(SEBlock(extra_conv_channels))
            self.block4: nn.Module = nn.Sequential(*layers)
            feature_dim = extra_conv_channels
        else:
            self.block4 = nn.Identity()
            feature_dim = 128

        # Global average pooling + classification head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        if direct_head:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(feature_dim, num_classes),
            )
        else:
            # new improvement: 2-layer MLP head preserving non-linear separation without parameter bloat
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(128, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, 3, H, W)``.

        Returns:
            Raw logit tensor of shape ``(B, num_classes)``.
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def parameter_count(self) -> dict[str, int]:
        """Return trainable and total parameter counts."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {"trainable": trainable, "total": total, "frozen": total - trainable}


# ─────────────────────────────────────────────────────────────────────────────
# Factory and I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_cnn_model(
    dropout: float = 0.4,
    use_extra_conv: bool = True,
    # new improvement: extra_conv_channels=128 avoids the 295k parameter spike
    extra_conv_channels: int = 128,
    num_classes: int = 3,
    use_se: bool = False,
    direct_head: bool = False,
) -> SeverityCNN:
    """Build and return a fresh :class:`SeverityCNN` (no pretrained weights).

    # new improvement: builds balanced architecture without Block 4 bottleneck
    """
    return SeverityCNN(
        num_classes=num_classes,
        dropout=dropout,
        use_extra_conv=use_extra_conv,
        extra_conv_channels=extra_conv_channels,
        use_se=use_se,
        direct_head=direct_head,
    )


def load_cnn_model(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> SeverityCNN:
    """Load a :class:`SeverityCNN` from a saved ``.pt`` checkpoint.

    Supports both improved checkpoints (with direct head and SE attention)
    and legacy checkpoints with automatic state dict shape detection.

    Args:
        checkpoint_path: Path to the ``.pt`` file.
        device: Target device (``"cpu"`` or ``"cuda"``).

    Returns:
        ``SeverityCNN`` in eval mode on the target device.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        KeyError: If ``model_state_dict`` key is missing.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Severity CNN checkpoint not found: {path}. "
            "Run notebooks/06_severity_cnn_training.ipynb to generate it."
        )

    checkpoint = torch.load(path, map_location=device, weights_only=True)

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint at {path} is missing 'model_state_dict'. "
            "Ensure it was saved with the standard ClaimVision checkpoint format."
        )

    state_dict = checkpoint["model_state_dict"]
    num_classes = checkpoint.get("num_classes", 3)
    dropout = checkpoint.get("dropout", 0.4)
    use_extra_conv = checkpoint.get("use_extra_conv", True)

    # new improvement: auto-detect Block 4 channel dimension (128 vs legacy 256)
    if "extra_conv_channels" in checkpoint:
        extra_conv_channels = int(checkpoint["extra_conv_channels"])
    elif "block4.0.weight" in state_dict:
        extra_conv_channels = state_dict["block4.0.weight"].shape[0]
    else:
        extra_conv_channels = 128

    # Auto-detect SEBlock presence if not explicitly stored
    if "use_se" in checkpoint:
        use_se = bool(checkpoint["use_se"])
    else:
        use_se = any("block4.3" in k or "fc.2.weight" in k for k in state_dict.keys())

    # Auto-detect direct head vs legacy 2-layer head
    if "direct_head" in checkpoint:
        direct_head = bool(checkpoint["direct_head"])
    elif "classifier.1.weight" in state_dict and "classifier.3.weight" not in state_dict:
        direct_head = True
    else:
        direct_head = False

    model = SeverityCNN(
        num_classes=num_classes,
        dropout=dropout,
        use_extra_conv=use_extra_conv,
        extra_conv_channels=extra_conv_channels,
        use_se=use_se,
        direct_head=direct_head,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def save_cnn_checkpoint(
    model: SeverityCNN,
    output_path: str | Path,
    extra: dict | None = None,
) -> None:
    """Save model state dict and metadata to a ``.pt`` file.

    Args:
        model: ``SeverityCNN`` instance.
        output_path: Destination path.
        extra: Optional extra metadata (e.g. epoch, val_f1).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "model_state_dict": model.state_dict(),
        "num_classes": model._num_classes,
        "dropout": model._dropout,
        "use_extra_conv": model._use_extra_conv,
        # new improvement: records extra_conv_channels for checkpoint reproducibility
        "extra_conv_channels": getattr(model, "_extra_conv_channels", 128),
        "use_se": getattr(model, "_use_se", False),
        "direct_head": getattr(model, "_direct_head", False),
        "model_class": "SeverityCNN",
        "experiment_id": "SEV-CNN-001",
    }
    if extra:
        payload.update(extra)
    torch.save(payload, output_path)


def export_onnx(
    model: SeverityCNN,
    output_path: str | Path,
    image_size: tuple[int, int] | int = CNN_IMAGE_SIZE_4_3,
    device: str = "cpu",
) -> None:
    """Export a :class:`SeverityCNN` to ONNX format (opset 17).

    The exported model accepts ``float32`` input of shape
    ``(B, 3, image_size, image_size)`` and returns logits ``(B, 3)``.

    Args:
        model: ``SeverityCNN`` in eval mode.
        output_path: Destination ``.onnx`` path.
        image_size: Spatial dimension used during export (must match training).
        device: Device for the dummy input.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # new improvement: supports both rectangular (H, W) and square dimensions
    h, w = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
    model.eval()
    dummy = torch.zeros(1, 3, h, w, device=device)

    try:
        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
            dynamo=False,
        )
    except TypeError:
        # Fallback for PyTorch < 2.2 where dynamo kwarg does not exist
        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        )


def save_cnn_preprocessing_config(
    output_path: str | Path,
    # new improvement: records native 4:3 image dimensions
    image_size: tuple[int, int] | int = CNN_IMAGE_SIZE_4_3,
) -> None:
    """Write the preprocessing configuration used at training time.

    # new improvement: records native 4:3 aspect ratio and edge-preserving settings
    This JSON is read by the unified pipeline (Phase 11) to ensure runtime
    preprocessing matches training preprocessing exactly.

    Args:
        output_path: Path to write the JSON file.
        image_size: Target image dimensions (default (192, 256)).
    """
    h, w = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
    config = {
        "model": "SeverityCNN",
        "experiment_id": "SEV-CNN-001",
        # new improvement: native 4:3 aspect ratio matching dataset median (194x259)
        "image_size": [h, w],
        "color_order": "RGB",
        "normalisation": {
            "mean": _IMAGENET_MEAN,
            "std": _IMAGENET_STD,
            "note": (
                "ImageNet statistics used for numeric stability. "
                "No ImageNet knowledge is transferred — this is a from-scratch model."
            ),
        },
        # new improvement: full-canvas resize without destructive center crop
        "resize_strategy": f"Resize(({h}, {w})) full-canvas for val/test (no crop)",
        # new improvement: streamlined edge-preserving augmentations
        "training_augmentations": [
            f"Resize(({h}, {w}))",
            "RandomHorizontalFlip(p=0.5)",
            "ColorJitter(brightness=0.1, contrast=0.1)",
        ],
        "class_map": SEVERITY_ID_TO_CLASS,
        "classes": SEVERITY_CLASSES,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)



# ─────────────────────────────────────────────────────────────────────────────
# Inference helper — usable by Phase 11 unified pipeline
# ─────────────────────────────────────────────────────────────────────────────

def predict_severity_cnn(
    image_path: str | Path,
    model: SeverityCNN,
    device: str = "cpu",
    image_size: tuple[int, int] | int = CNN_IMAGE_SIZE_4_3,
    class_weights: list[float] | None = None,
) -> dict:
    """Run single-image severity inference with the baseline CNN.

    Args:
        image_path: Path to a car damage image.
        model: ``SeverityCNN`` in eval mode.
        device: ``"cpu"`` or ``"cuda"``.
        image_size: Spatial dimension for inference transforms.
        class_weights: Optional list of 3 calibration multipliers [w_minor, w_moderate, w_severe].
            If provided, class selection uses argmax(probs * weights), enabling
            validation-calibrated decision boundaries without retrained weights.

    Returns:
        Dictionary with keys:
            ``predicted_class`` (str), ``predicted_id`` (int),
            ``probabilities`` (dict[str, float]),
            ``confidence`` (float), ``latency_ms`` (float),
            ``model_version`` (str), ``experiment_id`` (str).
    """
    transform = get_severity_transforms("test", image_size=image_size)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)  # (1, 3)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    if class_weights is not None:
        scores = probs * np.asarray(class_weights, dtype=np.float32)
        pred_id = int(np.argmax(scores))
    else:
        pred_id = int(np.argmax(probs))

    return {
        "predicted_class": SEVERITY_ID_TO_CLASS[pred_id],
        "predicted_id": pred_id,
        "probabilities": {
            cls: float(round(probs[i], 4))
            for i, cls in SEVERITY_ID_TO_CLASS.items()
        },
        "confidence": float(round(float(probs[pred_id]), 4)),
        "latency_ms": round(latency_ms, 2),
        "model_version": "SEV-CNN-001",
        "experiment_id": "SEV-CNN-001",
    }


# ─────────────────────────────────────────────────────────────────────────────
# CPU latency benchmark helper
# ─────────────────────────────────────────────────────────────────────────────

def measure_cpu_latency(
    model: SeverityCNN,
    image_size: tuple[int, int] | int = CNN_IMAGE_SIZE_4_3,
    n_warmup: int = 10,
    n_runs: int = 100,
) -> dict[str, float]:
    """Measure mean and std CPU inference latency (ms per image).

    Args:
        model: ``SeverityCNN`` in eval mode.
        image_size: Spatial dimension of dummy input (int or (H, W)).
        n_warmup: Number of warm-up forward passes (not timed).
        n_runs: Number of timed forward passes.

    Returns:
        ``{"mean_ms": float, "std_ms": float, "n_runs": int}``.
    """
    model.eval()
    h, w = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
    dummy = torch.zeros(1, 3, h, w)
    times = []

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            times.append((time.perf_counter() - t0) * 1000.0)

    return {
        "mean_ms": round(float(np.mean(times)), 2),
        "std_ms": round(float(np.std(times)), 2),
        "n_runs": n_runs,
    }
