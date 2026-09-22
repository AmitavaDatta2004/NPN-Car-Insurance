"""location/dataset.py — Dataset utilities for the damaged-part location classifier.

Task IDs : LOC-DATA-001
Phase    : 10b
Owner    : Detection ML member

Label derivation strategy
-------------------------
Each COCO image gets ONE label: the part class whose bounding boxes appear most
frequently in that image (dominant-part rule). On a tie, the part whose bounding
boxes have the greatest total area wins. This produces clean single-label data
from the COCO multi-box annotation format without requiring any new manual labels.

Scientific note
---------------
The COCO car damage dataset contains 59 train / 11 val / 8 test images spread
across 5 part classes. This yields approximately 12 images per class, which is
extremely small. ImageNet transfer learning (MobileNetV2 / EfficientNet-B0)
compensates, but generalization is limited. Document this in every notebook.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOCATION_CLASSES: list[str] = [
    "headlamp",
    "front_bumper",
    "hood",
    "door",
    "rear_bumper",
]

LOCATION_CLASS_TO_ID: dict[str, int] = {c: i for i, c in enumerate(LOCATION_CLASSES)}
LOCATION_ID_TO_CLASS: dict[int, str] = {i: c for i, c in enumerate(LOCATION_CLASSES)}

# ImageNet normalisation — transfer learning models expect this
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

LOCATION_IMAGE_SIZE: int = 224  # MobileNetV2 and EfficientNet-B0 standard input


# ---------------------------------------------------------------------------
# Label derivation
# ---------------------------------------------------------------------------

def _area(bbox: list[float]) -> float:
    """Return area of a COCO [x, y, w, h] bounding box."""
    return float(bbox[2]) * float(bbox[3])


def derive_location_labels(
    coco_json_path: str | Path,
    class_name_map: dict[str, str] | None = None,
) -> dict[str, tuple[int, str]]:
    """Derive a single dominant-part label per image from a COCO JSON file.

    Strategy
    --------
    For each image:
    1. Count how many bounding boxes belong to each part class.
    2. The class with the highest count is the label.
    3. On a tie, pick the class with the greatest total bounding-box area.
    4. Images with zero annotations for any of the 5 recognised classes are
       skipped (returned label = None; callers must filter these out).

    Parameters
    ----------
    coco_json_path:
        Path to a COCO annotation JSON with ``images``, ``annotations``, and
        ``categories`` keys.
    class_name_map:
        Optional mapping from raw category names in the JSON to canonical
        LOCATION_CLASSES names.  E.g. ``{"rear bumper": "rear_bumper"}``.
        If None, a default map is used that normalises common variants.

    Returns
    -------
    dict mapping ``image_id (str)`` → ``(label_id: int, label_name: str)``.
    Only images with at least one recognised-class annotation are returned.
    """
    coco_json_path = Path(coco_json_path)
    if not coco_json_path.exists():
        raise FileNotFoundError(f"COCO JSON not found: {coco_json_path}")

    with coco_json_path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    # Default name normalisation map
    _default_map: dict[str, str] = {
        "headlamp": "headlamp",
        "front_bumper": "front_bumper",
        "front bumper": "front_bumper",
        "hood": "hood",
        "door": "door",
        "rear_bumper": "rear_bumper",
        "rear bumper": "rear_bumper",
    }
    if class_name_map is None:
        class_name_map = {}
    merged_map = {**_default_map, **class_name_map}

    # Build category_id → canonical_class_name mapping
    cat_to_name: dict[int, str] = {}
    for cat in data.get("categories", []):
        raw_name = str(cat.get("name", "")).strip().lower()
        canonical = merged_map.get(raw_name)
        if canonical is not None:
            cat_to_name[cat["id"]] = canonical

    # Per-image: count boxes and sum areas per class
    img_counts: dict[int, Counter[str]] = defaultdict(Counter)
    img_areas: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for ann in data.get("annotations", []):
        cat_id = ann.get("category_id", -1)
        if cat_id not in cat_to_name:
            continue
        class_name = cat_to_name[cat_id]
        img_id = ann["image_id"]
        img_counts[img_id][class_name] += 1
        img_areas[img_id][class_name] += _area(ann.get("bbox", [0, 0, 0, 0]))

    # Resolve dominant part per image
    result: dict[str, tuple[int, str]] = {}
    for img in data.get("images", []):
        img_id = img["id"]
        fname = img.get("file_name", str(img_id))
        counts = img_counts.get(img_id)
        if not counts:
            continue  # no recognised-class annotations → skip

        max_count = max(counts.values())
        candidates = [cls for cls, cnt in counts.items() if cnt == max_count]

        if len(candidates) == 1:
            dominant = candidates[0]
        else:
            # Tiebreak by largest total bbox area
            areas = img_areas[img_id]
            dominant = max(candidates, key=lambda c: areas.get(c, 0.0))

        label_id = LOCATION_CLASS_TO_ID[dominant]
        result[fname] = (label_id, dominant)

    return result


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_location_transforms(
    split: Literal["train", "val", "test"],
    image_size: int = LOCATION_IMAGE_SIZE,
) -> transforms.Compose:
    """Return the augmentation pipeline for a given data split.

    Train:  random resized crop, horizontal flip, colour jitter, rotation.
    Val/test: deterministic resize + centre-crop.

    Conservative augmentation preserves surface texture and damage indicators.

    Parameters
    ----------
    split:
        ``"train"``, ``"val"``, or ``"test"``.
    image_size:
        Target square image side (default 224 for MobileNetV2 / EfficientNet).

    Returns
    -------
    ``transforms.Compose`` pipeline.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"Invalid split {split!r}. Must be 'train', 'val', or 'test'.")

    normalise = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
    resize_to = int(image_size * 256 / 224)

    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1, hue=0.05),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            normalise,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(resize_to),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalise,
        ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LocationDataset(Dataset):
    """PyTorch Dataset for damaged-part location image classification.

    Loads images from a COCO image directory and uses derive_location_labels
    to assign one dominant-part label per image.

    Parameters
    ----------
    image_dir:
        Directory containing the raw images (COCO train/ or val/).
    coco_json_path:
        COCO annotation JSON for this split (e.g. COCO_mul_train_annos.json).
    split:
        ``"train"``, ``"val"``, or ``"test"`` — controls augmentation.
    transform:
        Optional override transform. If None, uses get_location_transforms.
    image_size:
        Target square image side (default 224).
    class_name_map:
        Optional category name normalisation overrides.
    """

    def __init__(
        self,
        image_dir: str | Path | None = None,
        coco_json_path: str | Path | None = None,
        split: Literal["train", "val", "test"] = "train",
        transform: transforms.Compose | None = None,
        image_size: int = LOCATION_IMAGE_SIZE,
        class_name_map: dict[str, str] | None = None,
        samples: list[tuple[Path, int]] | None = None,
    ) -> None:
        self.image_dir = Path(image_dir) if image_dir is not None else None
        self.coco_json_path = Path(coco_json_path) if coco_json_path is not None else None
        self.split = split
        self.transform = (
            transform
            if transform is not None
            else get_location_transforms(split, image_size=image_size)
        )

        if samples is not None:
            self.samples = [(Path(p), int(l)) for p, l in samples if Path(p).exists()]
        else:
            if self.image_dir is None or self.coco_json_path is None:
                raise ValueError("Must provide either 'samples' or both 'image_dir' and 'coco_json_path'.")
            # Derive labels
            label_map = derive_location_labels(self.coco_json_path, class_name_map)

            # Build list of (abs_path, label_id) for images that exist on disk
            self.samples = []
            for fname, (label_id, _) in label_map.items():
                img_path = self.image_dir / Path(fname).name
                if img_path.exists():
                    self.samples.append((img_path, label_id))

        if not self.samples:
            source_desc = f"{self.image_dir} ({self.coco_json_path})" if samples is None else f"{len(samples)} provided samples"
            raise RuntimeError(
                f"No valid images found for location dataset from {source_desc}. "
                "Check that the COCO dataset is downloaded and paths are correct."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, label_id = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to open image {img_path}: {exc}") from exc
        image_tensor = self.transform(image)
        label_tensor = torch.tensor(label_id, dtype=torch.long)
        return image_tensor, label_tensor

    @property
    def label_counts(self) -> dict[int, int]:
        """Return ``{label_id: count}`` for all loaded samples."""
        counter: Counter[int] = Counter(lbl for _, lbl in self.samples)
        return dict(counter)

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights normalized so mean weight = 1.0."""
        total = len(self.samples)
        counts = self.label_counts
        weights = []
        for cls_id in range(len(LOCATION_CLASSES)):
            count = counts.get(cls_id, 1)
            weights.append(total / (len(LOCATION_CLASSES) * max(count, 1)))
        weights_t = torch.tensor(weights, dtype=torch.float32)
        return weights_t / weights_t.mean()

    def __repr__(self) -> str:
        counts = self.label_counts
        class_summary = ", ".join(
            f"{LOCATION_CLASSES[i]}={counts.get(i, 0)}" for i in range(len(LOCATION_CLASSES))
        )
        return f"LocationDataset(split={self.split!r}, n={len(self)}, {class_summary})"


# ---------------------------------------------------------------------------
# Split loader
# ---------------------------------------------------------------------------

def load_location_splits(
    raw_coco_dir: str | Path,
    val_ratio: float = 0.20,
    seed: int = 42,
    image_size: int = LOCATION_IMAGE_SIZE,
    class_name_map: dict[str, str] | None = None,
) -> tuple[LocationDataset, LocationDataset]:
    """Load train and validation LocationDatasets with robust handling for sparse val sets.

    If the ``val/`` directory contains >= 5 images on disk, folder-based splits are used.
    If ``val/`` has < 5 images on disk (such as in the truncated Kaggle archive where 10 of 11
    val images are missing from the folder), all available annotated images across train/ and val/
    are pooled, and a reproducible stratified train/val split is generated so that every class
    has validation samples and metrics are mathematically sound.

    Parameters
    ----------
    raw_coco_dir:
        Root directory containing ``train/`` and ``val/`` folders.
    val_ratio:
        Proportion of images to allocate to validation if pooling is needed (default 0.20).
    seed:
        Random seed for the stratified split (default 42).
    image_size:
        Target image dimension (default 224).
    class_name_map:
        Optional category mapping overrides.

    Returns
    -------
    tuple of (train_dataset, val_dataset)
    """
    raw_coco_dir = Path(raw_coco_dir)
    train_dir = raw_coco_dir / "train"
    val_dir = raw_coco_dir / "val"
    train_json = train_dir / "COCO_mul_train_annos.json"
    val_json = val_dir / "COCO_mul_val_annos.json"

    # 1. Collect all valid annotated samples from train/
    train_samples: list[tuple[Path, int]] = []
    if train_json.exists() and train_dir.exists():
        lmap_train = derive_location_labels(train_json, class_name_map)
        for fname, (lid, _) in lmap_train.items():
            ip = train_dir / Path(fname).name
            if ip.exists():
                train_samples.append((ip, lid))

    # 2. Collect all valid annotated samples from val/
    val_samples: list[tuple[Path, int]] = []
    if val_json.exists() and val_dir.exists():
        lmap_val = derive_location_labels(val_json, class_name_map)
        for fname, (lid, _) in lmap_val.items():
            ip = val_dir / Path(fname).name
            if ip.exists():
                val_samples.append((ip, lid))

    # 3. Check if val has enough physical images
    if len(val_samples) >= 5 and len(set(l for _, l in val_samples)) >= 3:
        # Sufficient validation samples in folder
        train_ds = LocationDataset(
            samples=train_samples, split="train", image_size=image_size
        )
        val_ds = LocationDataset(
            samples=val_samples, split="val", image_size=image_size
        )
        return train_ds, val_ds

    # 4. Fallback: pool all annotated images and perform stratified split
    all_samples = train_samples + val_samples
    if not all_samples:
        raise RuntimeError(
            f"No annotated images found in {train_dir} or {val_dir}. "
            "Please check raw dataset paths."
        )

    from sklearn.model_selection import train_test_split

    labels = [l for _, l in all_samples]
    label_counts = Counter(labels)
    n_classes = len(label_counts)

    # Calculate validation count; ensure at least n_classes when stratifying if enough samples
    val_count = max(int(round(val_ratio * len(all_samples))), n_classes)
    train_count = len(all_samples) - val_count

    can_stratify = (
        all(cnt >= 2 for cnt in label_counts.values())
        and val_count >= n_classes
        and train_count >= n_classes
    )

    stratify_arg = labels if can_stratify else None
    test_size_arg = val_count if can_stratify else val_ratio

    pooled_train, pooled_val = train_test_split(
        all_samples,
        test_size=test_size_arg,
        random_state=seed,
        stratify=stratify_arg,
    )

    train_ds = LocationDataset(
        samples=pooled_train, split="train", image_size=image_size
    )
    val_ds = LocationDataset(
        samples=pooled_val, split="val", image_size=image_size
    )

    return train_ds, val_ds
