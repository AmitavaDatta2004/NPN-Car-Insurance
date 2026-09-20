"""Fraud dataset auditing and shortcut risk analysis.

Provides CSV schema inspection, image-to-record reconciliation, exact hash
deduplication, perceptual hash clustering, and statistical shortcut detection.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import imagehash
import pandas as pd
from PIL import Image

from claimvision_ml.quality.image_checks import compute_image_metrics


def load_and_validate_csv(
    csv_path: str | Path,
    expected_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Load and validate dataset CSV schema.

    Args:
        csv_path: Path to dataset CSV file.
        expected_cols: Optional list of expected column names.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If CSV is empty or missing expected columns.
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset CSV not found at: {p}")

    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"Dataset CSV is empty: {p}")

    if expected_cols:
        missing = [col for col in expected_cols if col not in df.columns]
        if missing:
            raise ValueError(
                f"CSV missing expected columns: {missing}. Found: {list(df.columns)}"
            )

    return df


def audit_image_files(
    df: pd.DataFrame,
    images_dir: str | Path,
    filename_col: str = "filename",
) -> dict[str, Any]:
    """Reconcile CSV records with physical images on disk.

    Args:
        df: DataFrame containing dataset records.
        images_dir: Directory containing image files.
        filename_col: Column name specifying image filename.

    Returns:
        Dictionary with matched, missing, and orphan image lists.
    """
    img_dir = Path(images_dir)
    if not img_dir.exists() or not img_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {img_dir}")

    # Gather physical images (recursive search for supported extensions)
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    disk_files = {
        p.name: p
        for p in img_dir.rglob("*")
        if p.suffix.lower() in valid_exts and p.is_file()
    }

    csv_filenames = set(df[filename_col].dropna().astype(str).tolist())

    matched = []
    missing = []
    for fname in sorted(csv_filenames):
        clean_name = Path(fname).name
        if clean_name in disk_files:
            matched.append((fname, str(disk_files[clean_name])))
        else:
            missing.append(fname)

    matched_names = {Path(m[0]).name for m in matched}
    orphans = [
        str(path) for name, path in disk_files.items() if name not in matched_names
    ]

    return {
        "total_csv_records": len(df),
        "unique_csv_filenames": len(csv_filenames),
        "total_disk_images": len(disk_files),
        "matched_count": len(matched),
        "missing_count": len(missing),
        "orphan_count": len(orphans),
        "matched_pairs": matched,
        "missing_files": missing,
        "orphan_files": orphans,
    }


def compute_file_sha256(image_path: str | Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_file_hashes(image_paths: list[str | Path]) -> dict[str, str]:
    """Compute SHA-256 hashes for all provided image paths."""
    hashes = {}
    for p in image_paths:
        path_obj = Path(p)
        if path_obj.exists() and path_obj.is_file():
            hashes[str(path_obj)] = compute_file_sha256(path_obj)
    return hashes


def find_exact_duplicate_groups(file_hashes: dict[str, str]) -> list[list[str]]:
    """Group file paths by identical SHA-256 hash.

    Returns:
        List of duplicate groups (each group has 2 or more files).
    """
    hash_to_paths = defaultdict(list)
    for path, h in file_hashes.items():
        hash_to_paths[h].append(path)

    return [paths for paths in hash_to_paths.values() if len(paths) > 1]


def compute_perceptual_hashes(
    image_paths: list[str | Path],
    hash_size: int = 8,
    threshold: int = 4,
) -> tuple[dict[str, str], list[list[str]]]:
    """Compute perceptual difference hashes (dHash) and cluster near-duplicates.

    Args:
        image_paths: List of file paths to inspect.
        hash_size: Hash dimension (default 8 yields 64-bit hash).
        threshold: Max Hamming distance to consider two images near-duplicates.

    Returns:
        Tuple of (path_to_hash_str, list_of_near_duplicate_clusters).
    """
    path_to_hash: dict[str, imagehash.ImageHash] = {}
    hash_strs: dict[str, str] = {}

    for p in image_paths:
        try:
            with Image.open(p) as img:
                h = imagehash.dhash(img, hash_size=hash_size)
                path_to_hash[str(p)] = h
                hash_strs[str(p)] = str(h)
        except Exception:
            continue

    paths = list(path_to_hash.keys())
    visited = set()
    clusters: list[list[str]] = []

    for i in range(len(paths)):
        path_a = paths[i]
        if path_a in visited:
            continue
        cluster = [path_a]
        hash_a = path_to_hash[path_a]

        for j in range(i + 1, len(paths)):
            path_b = paths[j]
            if path_b in visited:
                continue
            hash_b = path_to_hash[path_b]
            if (hash_a - hash_b) <= threshold:
                cluster.append(path_b)
                visited.add(path_b)

        if len(cluster) > 1:
            clusters.append(cluster)
            visited.add(path_a)

    return hash_strs, clusters


def audit_image_quality_distribution(image_paths: list[str | Path]) -> pd.DataFrame:
    """Run OpenCV image metrics on all given paths and return DataFrame."""
    records = []
    for p in image_paths:
        metrics = compute_image_metrics(p)
        records.append(metrics)
    return pd.DataFrame(records)


def detect_shortcut_risks(
    df: pd.DataFrame,
    label_col: str = "label",
    metric_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Analyze statistical differences across classes to identify shortcut learning risks.

    Compares distributions of non-damage artifacts (resolution, blur, aspect ratio,
    brightness) across class labels.
    """
    if metric_cols is None:
        metric_cols = [
            "width",
            "height",
            "aspect_ratio",
            "blur_score",
            "brightness",
            "contrast",
        ]

    available_cols = [c for c in metric_cols if c in df.columns]
    summary_by_class = {}

    for label, group in df.groupby(label_col):
        summary_by_class[str(label)] = {
            col: {
                "mean": round(float(group[col].mean()), 4),
                "std": round(float(group[col].std()), 4),
                "median": round(float(group[col].median()), 4),
            }
            for col in available_cols
        }

    # Compute difference ratios / disparity for shortcut detection
    disparities = {}
    classes = list(summary_by_class.keys())
    if len(classes) >= 2:
        c1, c2 = classes[0], classes[1]
        for col in available_cols:
            m1 = summary_by_class[c1][col]["mean"]
            m2 = summary_by_class[c2][col]["mean"]
            diff = abs(m1 - m2)
            denom = max(abs(m1), abs(m2), 1e-6)
            relative_disparity = round(diff / denom, 4)
            disparities[col] = {
                "relative_disparity": relative_disparity,
                "shortcut_risk": "HIGH" if relative_disparity > 0.35 else "LOW",
            }

    return {
        "metrics_by_class": summary_by_class,
        "disparities": disparities,
    }
