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

# Default image size for this CNN (per README §10 Notebook 06)
CNN_IMAGE_SIZE: int = 160


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
    from claimvision_ml.severity.inputs import resolve_image

    return resolve_image(raw_path)


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
        image_size: Target square image size (default 160).
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: Literal["train", "val", "test"] = "train",
        transform: transforms.Compose | None = None,
        image_size: int = CNN_IMAGE_SIZE,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Severity manifest not found: {self.manifest_path}. "
                "Run notebooks/05_severity_dataset_audit.ipynb first."
            )

        df = pd.read_csv(self.manifest_path)

        # Validate required columns
        required = {"image_path", "label_id"}
        missing_cols = required - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Manifest missing required columns: {missing_cols}. "
                f"Found: {df.columns.tolist()}"
            )

        # Resolve paths cross-platform
        df["_resolved_path"] = df["image_path"].apply(
            lambda p: str(_resolve_severity_image_path(str(p)))
        )
        df["_exists"] = df["_resolved_path"].apply(lambda p: Path(p).is_file())

        self.df = df.reset_index(drop=True)
        self.split = split
        self.transform = (
            transform
            if transform is not None
            else get_severity_transforms(split, image_size=image_size)
        )

        # Cache label distribution for reporting
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
    image_size: int = CNN_IMAGE_SIZE,
) -> transforms.Compose:
    """Return the augmentation pipeline for a given data split.

    Train:  gentle augmentation — flip, crop, colour jitter, rotation.
    Val/test: deterministic resize + centre-crop only.

    ImageNet mean/std normalisation is applied in all splits.

    Note: Augmentation is kept conservative.  Scratches and surface texture
    are important signal; aggressive transforms risk destroying them.

    Args:
        split: ``"train"``, ``"val"``, or ``"test"``.
        image_size: Target square image side (default 160 for baseline CNN).

    Returns:
        ``transforms.Compose`` pipeline.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"Invalid split {split!r}. Must be 'train', 'val', or 'test'.")

    normalise = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
    resize_to = int(image_size * 256 / 224)  # scale to 180 for 160 target

    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            normalise,
        ])
    else:  # val or test — deterministic only
        return transforms.Compose([
            transforms.Resize(resize_to),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalise,
        ])


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class SeverityCNN(nn.Module):
    """Baseline severity classifier trained from scratch.

    4-block convolutional network (conv → bn → relu → pool) followed by
    global average pooling and a two-layer classification head.

    This model establishes a performance floor for the MobileNetV2 and
    ViT-Tiny transfer-learning experiments in Phases 6 and 7.

    Args:
        num_classes: Number of output classes (default 3).
        dropout: Dropout probability in the classifier head.
        use_extra_conv: If True, adds a 4th conv block (128→256 channels).
    """

    def __init__(
        self,
        num_classes: int = 3,
        dropout: float = 0.4,
        use_extra_conv: bool = True,
    ) -> None:
        super().__init__()
        self._num_classes = num_classes
        self._dropout = dropout
        self._use_extra_conv = use_extra_conv

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
        # Block 4 (optional) — 128 → 256
        if use_extra_conv:
            self.block4: nn.Module = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
            feature_dim = 256
        else:
            self.block4 = nn.Identity()
            feature_dim = 128

        # Global average pooling + classification head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
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
    num_classes: int = 3,
) -> SeverityCNN:
    """Build and return a fresh :class:`SeverityCNN` (no pretrained weights).

    Args:
        dropout: Dropout probability in the classifier head.
        use_extra_conv: Whether to include the 4th 128→256 conv block.
        num_classes: Number of output severity classes.

    Returns:
        ``SeverityCNN`` in training mode.
    """
    return SeverityCNN(num_classes=num_classes, dropout=dropout, use_extra_conv=use_extra_conv)


def load_cnn_model(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> SeverityCNN:
    """Load a :class:`SeverityCNN` from a saved ``.pt`` checkpoint.

    The checkpoint must have been saved with::

        torch.save({
            "model_state_dict": model.state_dict(),
            "num_classes": 3,
            "dropout": 0.4,
            "use_extra_conv": True,
            ...
        }, path)

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

    num_classes = checkpoint.get("num_classes", 3)
    dropout = checkpoint.get("dropout", 0.4)
    use_extra_conv = checkpoint.get("use_extra_conv", True)

    model = SeverityCNN(
        num_classes=num_classes,
        dropout=dropout,
        use_extra_conv=use_extra_conv,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
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
        "model_class": "SeverityCNN",
        "experiment_id": "SEV-CNN-001",
    }
    if extra:
        payload.update(extra)
    torch.save(payload, output_path)


def export_onnx(
    model: SeverityCNN,
    output_path: str | Path,
    image_size: int = CNN_IMAGE_SIZE,
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

    model.eval()
    dummy = torch.zeros(1, 3, image_size, image_size, device=device)

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


def save_cnn_preprocessing_config(output_path: str | Path) -> None:
    """Write the preprocessing configuration used at training time.

    This JSON is read by the unified pipeline (Phase 11) to ensure runtime
    preprocessing matches training preprocessing exactly.

    Args:
        output_path: Path to write the JSON file.
    """
    config = {
        "model": "SeverityCNN",
        "experiment_id": "SEV-CNN-001",
        "image_size": [CNN_IMAGE_SIZE, CNN_IMAGE_SIZE],
        "color_order": "RGB",
        "normalisation": {
            "mean": _IMAGENET_MEAN,
            "std": _IMAGENET_STD,
            "note": (
                "ImageNet statistics used for numeric stability. "
                "No ImageNet knowledge is transferred — this is a from-scratch model."
            ),
        },
        "resize_strategy": f"Resize({int(CNN_IMAGE_SIZE * 256 / 224)}) + CenterCrop({CNN_IMAGE_SIZE}) for val/test",
        "training_augmentations": [
            f"RandomResizedCrop({CNN_IMAGE_SIZE}, scale=(0.75, 1.0))",
            "RandomHorizontalFlip(p=0.5)",
            "ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)",
            "RandomRotation(degrees=15)",
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
    image_size: int = CNN_IMAGE_SIZE,
) -> dict:
    """Run single-image severity inference with the baseline CNN.

    Args:
        image_path: Path to a car damage image.
        model: ``SeverityCNN`` in eval mode.
        device: ``"cpu"`` or ``"cuda"``.
        image_size: Spatial dimension for inference transforms.

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
    image_size: int = CNN_IMAGE_SIZE,
    n_warmup: int = 10,
    n_runs: int = 100,
) -> dict[str, float]:
    """Measure mean and std CPU inference latency (ms per image).

    Args:
        model: ``SeverityCNN`` in eval mode.
        image_size: Spatial dimension of dummy input.
        n_warmup: Number of warm-up forward passes (not timed).
        n_runs: Number of timed forward passes.

    Returns:
        ``{"mean_ms": float, "std_ms": float, "n_runs": int}``.
    """
    model.eval()
    dummy = torch.zeros(1, 3, image_size, image_size)
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
