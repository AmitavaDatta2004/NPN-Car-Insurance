"""PyTorch Dataset and transforms for ClaimVision AI Severity Classification.

Reads frozen manifests (severity_train.csv, severity_val.csv, severity_test.csv)
and provides bimodal path resolution for local and Colab environments.
"""

from __future__ import annotations

import csv
import logging
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

logger = logging.getLogger(__name__)

SEVERITY_CLASSES: tuple[str, ...] = ("minor", "moderate", "severe")
SEVERITY_CLASS_TO_ID: dict[str, int] = {cls_name: i for i, cls_name in enumerate(SEVERITY_CLASSES)}
SEVERITY_ID_TO_CLASS: dict[int, str] = {i: cls_name for i, cls_name in enumerate(SEVERITY_CLASSES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_MEAN_RGB = tuple(round(c * 255) for c in IMAGENET_MEAN)


def resolve_image_path(path_str: str, base_dirs: list[Path] | None = None) -> Path:
    """Resolve an image path across local, repository, and Colab environments."""
    p = Path(path_str)
    if p.is_file():
        return p

    search_roots = list(base_dirs) if base_dirs else []
    default_roots = [
        Path.cwd(),
        Path(__file__).resolve().parents[4],  # repo root
        Path("/content/NPN-Car-Insurance"),
        Path("/content"),
        Path.home() / ".cache" / "kagglehub",
        Path("/root/.cache/kagglehub"),
    ]
    for r in default_roots:
        if r not in search_roots:
            search_roots.append(r)

    # Try relative to search roots
    for root in search_roots:
        candidate = root / path_str
        if candidate.is_file():
            return candidate

    # Try matching filename inside car_damage_severity or data3a
    filename = p.name
    for root in search_roots:
        if root.exists():
            matches = list(root.glob(f"**/car_damage_severity/**/{filename}"))
            if matches:
                return matches[0]
            matches_d3a = list(root.glob(f"**/data3a/**/{filename}"))
            if matches_d3a:
                return matches_d3a[0]

    return p


class SeverityDataset(Dataset):
    """PyTorch Dataset loading vehicle damage severity images from CSV manifests.

    Args:
        manifest_path: Path to the frozen manifest CSV.
        transform: Optional torchvision transform to apply to PIL Image.
        base_dir: Optional root directory for relative path resolution.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        transform: Callable | None = None,
        base_dir: str | Path | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.transform = transform
        self.base_dir = Path(base_dir) if base_dir else None

        if not self.manifest_path.is_file():
            # Try searching repo root
            repo_root = Path(__file__).resolve().parents[4]
            candidate = repo_root / self.manifest_path
            if candidate.is_file():
                self.manifest_path = candidate
            else:
                raise FileNotFoundError(f"Severity manifest not found: {manifest_path}")

        self.records: list[dict[str, Any]] = []
        with open(self.manifest_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row["label"].strip().lower()
                label_id = int(row.get("label_id", SEVERITY_CLASS_TO_ID.get(label, -1)))
                if label_id not in (0, 1, 2):
                    raise ValueError(f"Invalid label '{label}' / id {label_id} in {self.manifest_path}")
                self.records.append(
                    {
                        "image_path": row["image_path"],
                        "filename": row.get("filename", Path(row["image_path"]).name),
                        "label": label,
                        "label_id": label_id,
                        "split": row.get("split", ""),
                        "blur_score": float(row.get("blur_score", 0.0)),
                        "brightness": float(row.get("brightness", 0.0)),
                        "contrast": float(row.get("contrast", 0.0)),
                    }
                )

        if not self.records:
            raise ValueError(f"Manifest {self.manifest_path} contains zero records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, dict[str, Any]]:
        record = self.records[idx]
        raw_path = record["image_path"]

        search_roots = [self.base_dir] if self.base_dir else []
        resolved_path = resolve_image_path(raw_path, search_roots)

        if not resolved_path.is_file():
            raise FileNotFoundError(f"Image not found at {raw_path} or resolved {resolved_path}")

        # Decode via OpenCV
        img_bgr = cv2.imread(str(resolved_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Corrupt or unreadable image: {resolved_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        if self.transform is not None:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = transforms.ToTensor()(img_pil)

        return img_tensor, record["label_id"], record

    def get_labels(self) -> list[int]:
        """Return list of integer labels for all records in the dataset."""
        return [r["label_id"] for r in self.records]

    def get_class_counts(self) -> dict[str, int]:
        """Return counts of each severity class."""
        counts = {c: 0 for c in SEVERITY_CLASSES}
        for r in self.records:
            counts[r["label"]] += 1
        return counts


def get_severity_transforms(split: str = "train", img_size: int = 224) -> transforms.Compose:
    """Return torchvision image transformations for severity classification.

    Full-canvas field of view is preserved without destructive CenterCrop,
    retaining all corner and bumper collision damage features.

    Args:
        split: One of 'train', 'val', or 'test'.
        img_size: Target square image dimension (default 224 for MobileNetV2).

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
                transforms.RandomAffine(degrees=8, translate=(0.04, 0.04)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    else:
        # val or test — preserve full vehicle canvas
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )


class LetterboxResize:
    """Aspect-ratio preserving resize with symmetrical ImageNet-mean padding."""

    def __init__(
        self,
        size: int,
        fill: tuple[int, int, int] = IMAGENET_MEAN_RGB,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ) -> None:
        self.size = int(size)
        self.fill = tuple(fill)
        self.interpolation = interpolation

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size

        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions: {width}x{height}")

        scale = self.size / max(width, height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))

        image = TF.resize(
            image,
            [resized_height, resized_width],
            interpolation=self.interpolation,
            antialias=True,
        )

        horizontal_padding = self.size - resized_width
        vertical_padding = self.size - resized_height

        left = horizontal_padding // 2
        right = horizontal_padding - left
        top = vertical_padding // 2
        bottom = vertical_padding - top

        image = TF.pad(
            image,
            [left, top, right, bottom],
            fill=self.fill,
        )
        return image

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"size={self.size}, "
            f"fill={self.fill}, "
            f"interpolation={self.interpolation})"
        )


def build_letterbox_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    """Build training and evaluation transforms using aspect-ratio preserving letterbox padding.

    Args:
        image_size: Target square canvas size (e.g. 224, 288, 320).

    Returns:
        (train_transform, eval_transform)
    """
    letterbox = LetterboxResize(
        size=image_size,
        fill=IMAGENET_MEAN_RGB,
        interpolation=InterpolationMode.BILINEAR,
    )

    train_transform = transforms.Compose(
        [
            letterbox,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.15,
            ),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )

    eval_transform = transforms.Compose(
        [
            LetterboxResize(
                size=image_size,
                fill=IMAGENET_MEAN_RGB,
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )

    return train_transform, eval_transform


def get_severity_class_weights(manifest_path: str | Path) -> torch.Tensor:
    """Compute balanced class weights for CrossEntropyLoss from manifest.

    weight_c = N / (n_classes * N_c)
    """
    dataset = SeverityDataset(manifest_path)
    counts = dataset.get_class_counts()
    total = len(dataset)
    n_classes = len(SEVERITY_CLASSES)

    weights = [
        total / (n_classes * max(counts[c], 1))
        for c in SEVERITY_CLASSES
    ]
    tensor_w = torch.tensor(weights, dtype=torch.float32)
    # Normalize so sum equals n_classes
    return tensor_w / tensor_w.mean()


def seed_worker(worker_id: int) -> None:
    """Seed individual DataLoader worker processes deterministically."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_severity_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = False,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Create a DataLoader with dedicated generator and worker seeding for reproducible training.

    Args:
        dataset: PyTorch dataset.
        batch_size: Mini-batch size.
        shuffle: Whether to shuffle data per epoch.
        seed: Random seed for dedicated generator.
        num_workers: Number of worker subprocesses.
        pin_memory: Whether to copy tensors into CUDA pinned memory.

    Returns:
        Configured DataLoader.
    """
    generator = None
    worker_init = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
        worker_init = seed_worker if num_workers > 0 else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        worker_init_fn=worker_init,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
