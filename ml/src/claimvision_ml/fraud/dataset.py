"""PyTorch Dataset and transform utilities for the fraud classifier.

Reads manifest CSVs produced by Phase 1 (01_fraud_dataset_audit.ipynb).
Each row in the manifest has an absolute `path` column and an integer `label`
column (0=genuine, 1=suspicious).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ImageNet normalisation constants
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class FraudDataset(Dataset):
    """Dataset that reads a fraud manifest CSV and returns (image_tensor, label_tensor).

    Args:
        manifest_path: Path to a CSV file with at minimum columns `path` and `label`.
        split: One of "train", "val", or "test". Controls which transforms are applied.
        transform: Optional override transform. If None, uses get_transforms(split).
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: Literal["train", "val", "test"] = "train",
        transform: transforms.Compose | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}. "
                "Run 01_fraud_dataset_audit.ipynb first."
            )

        df = pd.read_csv(self.manifest_path)
        required = {"path", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Manifest is missing required columns: {missing}. "
                f"Found: {df.columns.tolist()}"
            )

        # Keep only rows whose image file actually exists
        df["_exists"] = df["path"].apply(lambda p: Path(p).is_file())
        missing_count = (~df["_exists"]).sum()
        if missing_count > 0:
            import warnings
            warnings.warn(
                f"{missing_count} image paths in manifest do not exist on disk "
                "and will be skipped.",
                stacklevel=2,
            )
        self.df = df[df["_exists"]].reset_index(drop=True)

        self.split = split
        self.transform = transform if transform is not None else get_transforms(split)

        # Cache label distribution for reporting
        self._label_counts = self.df["label"].value_counts().to_dict()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        image_path = Path(row["path"])

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to open image {image_path}: {exc}"
            ) from exc

        image_tensor = self.transform(image)
        label_tensor = torch.tensor(float(row["label"]), dtype=torch.float32)
        return image_tensor, label_tensor

    @property
    def label_counts(self) -> dict[int, int]:
        """Return {label_int: count} for the loaded manifest."""
        return self._label_counts

    def __repr__(self) -> str:
        counts = self._label_counts
        return (
            f"FraudDataset(split={self.split!r}, "
            f"n={len(self)}, "
            f"genuine={counts.get(0, 0)}, "
            f"suspicious={counts.get(1, 0)})"
        )


def get_class_weights(manifest_path: str | Path) -> torch.Tensor:
    """Calculate inverse-frequency class weights for BCEWithLogitsLoss pos_weight.

    Returns a single-element tensor [w_positive] where w_positive is
    n_genuine / n_suspicious. This is the `pos_weight` argument to
    BCEWithLogitsLoss which up-weights the minority (suspicious) class.

    Args:
        manifest_path: Path to any split manifest CSV with a `label` column.

    Returns:
        Tensor of shape (1,) with the positive class weight.
    """
    df = pd.read_csv(manifest_path)
    counts = df["label"].value_counts()
    n_genuine = int(counts.get(0, 1))
    n_suspicious = int(counts.get(1, 1))
    pos_weight = n_genuine / n_suspicious
    return torch.tensor([pos_weight], dtype=torch.float32)


def get_transforms(
    split: Literal["train", "val", "test"],
) -> transforms.Compose:
    if split not in ("train", "val", "test"):
        raise ValueError(
            f"Invalid split {split!r}. Must be one of 'train', 'val', 'test'."
        )
    """Return the appropriate torchvision transform pipeline for a given split.

    Train transforms apply light CPU-friendly augmentation.
    Val/test transforms apply only deterministic resizing and cropping.

    ImageNet mean/std normalisation is applied in all splits.

    Args:
        split: "train", "val", or "test".

    Returns:
        transforms.Compose pipeline.
    """
    normalise = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            normalise,
        ])
    else:  # val or test — deterministic
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalise,
        ])
