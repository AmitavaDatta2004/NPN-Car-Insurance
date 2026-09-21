"""PyTorch Dataset and transform utilities for the fraud classifier.

Reads manifest CSVs produced by Phase 1 (01_fraud_dataset_audit.ipynb).
Each row in the manifest has an absolute `path` column and an integer `label`
column (0=genuine, 1=suspicious).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

# ImageNet normalisation constants
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _resolve_image_path(raw_path: str) -> Path:
    """Resolve image path across Windows, Linux, Colab, and local environments."""
    p = Path(raw_path)
    if p.is_file():
        return p
    norm = str(raw_path).replace("\\", "/")
    for marker in ["data/raw/", "data/samples/", "data/"]:
        if marker in norm:
            rel = norm[norm.index(marker):]
            for base in [Path.cwd(), Path("/content/NPN-Car-Insurance"), Path.cwd().parent]:
                candidate = (base / rel).resolve()
                if candidate.is_file():
                    return candidate
                candidate_nb = (base / "notebooks" / rel).resolve()
                if candidate_nb.is_file():
                    return candidate_nb
    fname = Path(norm).name
    for img_dir in [
        Path("data/raw/vinayjose_car_damage/images"),
        Path("/content/NPN-Car-Insurance/data/raw/vinayjose_car_damage/images"),
        Path("notebooks/data/raw/vinayjose_car_damage/images"),
    ]:
        candidate = (img_dir / fname).resolve()
        if candidate.is_file():
            return candidate
    return p


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

        # Resolve image paths cross-platform (handles Colab vs Windows differences)
        df["path"] = df["path"].apply(lambda p: str(_resolve_image_path(p)))
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


class BalancedEpochSampler(Sampler):
    """Per-epoch random undersampling sampler for the fraud classifier.

    Every epoch this sampler yields:
    - ALL suspicious (class 1) dataset indices  — fixed, same every epoch
    - A fresh random subset of N genuine (class 0) indices — changes each epoch

    The ratio ``suspicious_pct`` controls N:

        suspicious_pct = 0.5  →  50:50  →  N = n_susp      (e.g. 325 gen)
        suspicious_pct = 0.4  →  40:60  →  N = n_susp * 1.5 (e.g. 487 gen)
        suspicious_pct = 0.3  →  30:70  →  N = n_susp * 7/3 (e.g. 758 gen)
        suspicious_pct = 0.2  →  20:80  →  N = n_susp * 4   (e.g. 1300 gen)

    The per-epoch genuine sample is drawn WITHOUT replacement within each epoch.
    Across epochs a different random seed is used (base_seed + epoch_counter),
    so the model sees different genuine images in each epoch.

    Loss function should be plain ``BCEWithLogitsLoss()`` (no pos_weight) because
    the classes are already balanced by sampling.

    Args:
        dataset: A ``FraudDataset`` instance.  Its ``.df`` must have a ``label`` column
            with integer values 0 (genuine) and 1 (suspicious).
        suspicious_pct: Fraction of each epoch that should be suspicious images.
            Must be in (0, 1].  E.g. 0.5 for 50:50, 0.4 for 40:60.
        seed: Base random seed.  Epoch i uses seed ``seed + i``.

    Example::

        dataset = FraudDataset("fraud_train.csv", split="train")
        sampler = BalancedEpochSampler(dataset, suspicious_pct=0.5, seed=42)
        loader  = DataLoader(dataset, batch_size=16, sampler=sampler)
        for epoch in range(30):
            for images, labels in loader:   # sampler rebuilds each epoch
                ...
    """

    def __init__(
        self,
        dataset: "FraudDataset",
        suspicious_pct: float = 0.5,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if not (0.0 < suspicious_pct <= 1.0):
            raise ValueError(
                f"suspicious_pct must be in (0, 1], got {suspicious_pct}"
            )

        self._seed = seed
        self._suspicious_pct = suspicious_pct
        self._epoch: int = 0

        labels = dataset.df["label"].values  # numpy array
        self._susp_idx: np.ndarray = np.where(labels == 1)[0]
        self._gen_idx: np.ndarray = np.where(labels == 0)[0]

        n_susp = len(self._susp_idx)
        if n_susp == 0:
            raise ValueError("Dataset contains no suspicious (label=1) images.")
        if len(self._gen_idx) == 0:
            raise ValueError("Dataset contains no genuine (label=0) images.")

        # Number of genuine images to sample per epoch
        self._n_gen_per_epoch: int = max(
            1, round(n_susp * (1.0 - suspicious_pct) / suspicious_pct)
        )
        if self._n_gen_per_epoch > len(self._gen_idx):
            import warnings
            warnings.warn(
                f"BalancedEpochSampler: n_genuine_per_epoch ({self._n_gen_per_epoch}) "
                f"exceeds genuine pool ({len(self._gen_idx)}). "
                "Capping at the full genuine pool.",
                stacklevel=2,
            )
            self._n_gen_per_epoch = len(self._gen_idx)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def epoch(self) -> int:
        """Current epoch counter (increments every time __iter__ is called)."""
        return self._epoch

    @property
    def n_genuine_per_epoch(self) -> int:
        """Number of genuine images sampled per epoch."""
        return self._n_gen_per_epoch

    @property
    def epoch_size(self) -> int:
        """Total number of indices yielded per epoch."""
        return len(self._susp_idx) + self._n_gen_per_epoch

    # ------------------------------------------------------------------
    # Sampler protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.epoch_size

    def __iter__(self):
        rng = np.random.default_rng(self._seed + self._epoch)
        gen_sample = rng.choice(
            self._gen_idx, size=self._n_gen_per_epoch, replace=False
        )
        # Suspicious indices: always the same fixed set
        combined = np.concatenate([self._susp_idx, gen_sample])
        rng.shuffle(combined)
        self._epoch += 1
        return iter(combined.tolist())

    def __repr__(self) -> str:
        return (
            f"BalancedEpochSampler("
            f"suspicious_pct={self._suspicious_pct}, "
            f"n_susp={len(self._susp_idx)}, "
            f"n_gen_per_epoch={self._n_gen_per_epoch}, "
            f"epoch_size={self.epoch_size}, "
            f"epoch={self._epoch})"
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
