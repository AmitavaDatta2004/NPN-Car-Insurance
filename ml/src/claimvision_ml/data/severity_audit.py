"""Severity dataset auditing, duplicate detection, and manifest management.

Audits the 3-class Car Damage Severity Dataset (minor, moderate, severe),
detects corrupt files, exact SHA-256 duplicates, and perceptual hash clusters,
analyzes quality distributions, and produces frozen, duplicate-safe 70/15/15
stratified train/val/test manifests.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from claimvision_ml.quality.image_checks import (
    calculate_blur_score,
    calculate_brightness,
    calculate_contrast,
    read_image_safely,
)

SEVERITY_CLASSES: list[str] = ["minor", "moderate", "severe"]
SEVERITY_CLASS_TO_ID: dict[str, int] = {"minor": 0, "moderate": 1, "severe": 2}
SEVERITY_ID_TO_CLASS: dict[int, str] = {0: "minor", 1: "moderate", 2: "severe"}


def normalize_class_name(raw_name: str) -> str:
    """Normalize raw folder names to standard severity class names.

    Maps:
        '01-minor', 'minor', '1' -> 'minor'
        '02-moderate', 'moderate', '2' -> 'moderate'
        '03-severe', 'severe', '3' -> 'severe'
    """
    clean = raw_name.strip().lower()
    if "minor" in clean or clean == "01" or clean == "1":
        return "minor"
    if "moderate" in clean or clean == "02" or clean == "2":
        return "moderate"
    if "severe" in clean or clean == "03" or clean == "3":
        return "severe"
    raise ValueError(f"Unrecognized severity class folder name: {raw_name}")


def discover_severity_images(dataset_dir: str | Path) -> pd.DataFrame:
    """Scan dataset directory and collect image paths, classes, and original splits.

    Supports multiple common directory layouts:
      1. Standard structure: `<root>/data3a/training/01-minor/`, etc.
      2. Direct split folders: `<root>/training/01-minor/`, `<root>/validation/01-minor/`
      3. Flat split folders: `<root>/train/minor/`, `<root>/val/minor/`

    Returns:
        DataFrame with columns:
          - image_path: normalized repository-relative or absolute POSIX path string
          - filename: image file name
          - label: 'minor' | 'moderate' | 'severe'
          - label_id: 0 | 1 | 2
          - original_split: 'training' | 'validation' | 'unknown'
    """
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Severity dataset directory not found: {root}")

    # Identify search root (check for nested data3a directory if present)
    search_root = root / "data3a" if (root / "data3a").is_dir() else root

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    records = []

    for file_path in sorted(search_root.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in valid_exts:
            continue

        # Extract split and class from path parts
        parts = file_path.relative_to(search_root).parts
        if len(parts) >= 2:
            raw_split = parts[0].lower()
            raw_class = parts[1]
        else:
            raw_split = "unknown"
            raw_class = parts[0]

        try:
            label = normalize_class_name(raw_class)
        except ValueError:
            continue

        original_split = (
            "training"
            if "train" in raw_split
            else ("validation" if "val" in raw_split else "unknown")
        )

        records.append(
            {
                "image_path": file_path.as_posix(),
                "filename": file_path.name,
                "label": label,
                "label_id": SEVERITY_CLASS_TO_ID[label],
                "original_split": original_split,
            }
        )

    if not records:
        raise ValueError(f"No valid severity images found in: {dataset_dir}")

    df = pd.DataFrame(records)
    return df.sort_values("image_path").reset_index(drop=True)


def compute_file_sha256(file_path: str | Path) -> str:
    """Compute SHA-256 hexadecimal digest of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_image_phash(file_path: str | Path, hash_size: int = 8) -> str | None:
    """Compute perceptual difference hash (pHash) for an image.

    Returns:
        Hexadecimal hash string, or None if image cannot be decoded.
    """
    try:
        with Image.open(file_path) as img:
            return str(imagehash.phash(img, hash_size=hash_size))
    except Exception:
        return None


def audit_severity_images(
    df: pd.DataFrame,
    path_col: str = "image_path",
    label_col: str = "label",
) -> pd.DataFrame:
    """Validate all images, compute image metrics, SHA-256 digests, and pHash values.

    Args:
        df: DataFrame containing image records.
        path_col: Column with image paths.
        label_col: Column with class labels.

    Returns:
        DataFrame enriched with:
          - is_corrupt: bool (True if decode failed)
          - width, height, channels, aspect_ratio
          - blur_score: Laplacian variance
          - brightness: mean grayscale value
          - contrast: std dev of grayscale value
          - sha256: SHA-256 hexadecimal string
          - phash: 64-bit perceptual hash string
    """
    rows = []
    for _, row in df.iterrows():
        p = Path(row[path_col])
        item = dict(row)

        if not p.exists() or not p.is_file():
            item.update(
                {
                    "is_corrupt": True,
                    "width": 0,
                    "height": 0,
                    "channels": 0,
                    "aspect_ratio": 0.0,
                    "blur_score": 0.0,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "sha256": "",
                    "phash": "",
                }
            )
            rows.append(item)
            continue

        # File hash
        sha256 = compute_file_sha256(p)
        phash = compute_image_phash(p)

        # Image decode check via OpenCV
        img_bgr, _ = read_image_safely(p)
        if img_bgr is None:
            item.update(
                {
                    "is_corrupt": True,
                    "width": 0,
                    "height": 0,
                    "channels": 0,
                    "aspect_ratio": 0.0,
                    "blur_score": 0.0,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "sha256": sha256,
                    "phash": phash or "",
                }
            )
            rows.append(item)
            continue

        h, w = img_bgr.shape[:2]
        c = img_bgr.shape[2] if img_bgr.ndim == 3 else 1
        aspect_ratio = round(w / max(h, 1), 4)
        blur_score = round(calculate_blur_score(img_bgr), 2)
        brightness = round(calculate_brightness(img_bgr), 2)
        contrast = round(calculate_contrast(img_bgr), 2)

        item.update(
            {
                "is_corrupt": False,
                "width": w,
                "height": h,
                "channels": c,
                "aspect_ratio": aspect_ratio,
                "blur_score": blur_score,
                "brightness": brightness,
                "contrast": contrast,
                "sha256": sha256,
                "phash": phash or "",
            }
        )
        rows.append(item)

    return pd.DataFrame(rows)


def find_exact_duplicate_groups(audit_df: pd.DataFrame) -> dict[str, list[str]]:
    """Identify exact SHA-256 duplicate image groups.

    Returns:
        Dictionary mapping SHA-256 hash to list of duplicate image paths.
    """
    valid = audit_df[~audit_df["is_corrupt"]]
    hash_map: dict[str, list[str]] = defaultdict(list)
    for _, row in valid.iterrows():
        hash_map[row["sha256"]].append(str(row["image_path"]))

    return {h: paths for h, paths in hash_map.items() if len(paths) > 1}


def find_perceptual_duplicates(
    audit_df: pd.DataFrame,
    threshold: int = 8,
) -> list[list[str]]:
    """Cluster near-duplicate images using pHash Hamming distance.

    Args:
        audit_df: Audited DataFrame with 'phash' and 'image_path' columns.
        threshold: Maximum bit Hamming distance to group into cluster.

    Returns:
        List of clusters, each cluster containing >= 2 image paths.
    """
    valid = audit_df[~audit_df["is_corrupt"] & (audit_df["phash"] != "")]
    parsed_hashes: list[tuple[str, imagehash.ImageHash]] = []

    for _, row in valid.iterrows():
        try:
            h = imagehash.hex_to_hash(row["phash"])
            parsed_hashes.append((str(row["image_path"]), h))
        except Exception:
            continue

    clusters: list[list[str]] = []
    visited: set[str] = set()

    for i in range(len(parsed_hashes)):
        path_a, hash_a = parsed_hashes[i]
        if path_a in visited:
            continue

        cluster = [path_a]
        for j in range(i + 1, len(parsed_hashes)):
            path_b, hash_b = parsed_hashes[j]
            if path_b in visited:
                continue

            dist = hash_a - hash_b
            if dist <= threshold:
                cluster.append(path_b)
                visited.add(path_b)

        if len(cluster) > 1:
            clusters.append(cluster)
            visited.add(path_a)

    return clusters


def create_severity_split(
    audit_df: pd.DataFrame,
    duplicate_clusters: list[list[str]] | None = None,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a stratified, duplicate-safe train/val/test split.

    Guarantees:
      1. Zero corrupt images in any split.
      2. No exact or near-duplicate pair crosses split boundaries.
      3. Class balance preserved across train, val, and test.

    Returns:
        (train_df, val_df, test_df)
    """
    if abs(sum(ratios) - 1.0) > 1e-4:
        raise ValueError(f"Ratios must sum to 1.0, got: {ratios}")

    # Exclude corrupt images
    valid_df = audit_df[~audit_df["is_corrupt"]].copy()

    # Assign each image a cluster ID (independent by default)
    path_to_cluster: dict[str, str] = {
        str(row["image_path"]): f"item_{idx}"
        for idx, (_, row) in enumerate(valid_df.iterrows())
    }

    # Group exact duplicates first
    exact_groups = find_exact_duplicate_groups(valid_df)
    for group_paths in exact_groups.values():
        canonical = min(path_to_cluster[p] for p in group_paths if p in path_to_cluster)
        for p in group_paths:
            if p in path_to_cluster:
                path_to_cluster[p] = canonical

    # Merge near-duplicate clusters
    if duplicate_clusters:
        for cluster in duplicate_clusters:
            existing_cids = {
                path_to_cluster[p] for p in cluster if p in path_to_cluster
            }
            if len(existing_cids) > 1:
                canonical = min(existing_cids)
                for p in cluster:
                    if p in path_to_cluster:
                        path_to_cluster[p] = canonical

    valid_df["_cluster_id"] = valid_df["image_path"].astype(str).map(path_to_cluster)

    # Summarize clusters by majority label and size
    cluster_summaries = []
    for cid, group in valid_df.groupby("_cluster_id"):
        mode_label = group["label"].mode().iloc[0]
        cluster_summaries.append(
            {
                "_cluster_id": cid,
                "label": mode_label,
                "size": len(group),
                "paths": list(group["image_path"]),
            }
        )

    cluster_df = pd.DataFrame(cluster_summaries)
    rng = np.random.RandomState(seed)

    train_cids: set[str] = set()
    val_cids: set[str] = set()
    test_cids: set[str] = set()

    # Stratify by class across clusters
    for label, group in cluster_df.groupby("label"):
        shuffled = group.sample(frac=1.0, random_state=rng)
        total_items = shuffled["size"].sum()
        target_train = total_items * ratios[0]
        target_val = total_items * ratios[1]

        curr_train = 0
        curr_val = 0

        rows = list(shuffled.iterrows())
        if len(rows) >= 3 and ratios[0] > 0 and ratios[1] > 0 and ratios[2] > 0:
            # Guarantee at least 1 cluster per split for minority classes/small datasets
            _, r_val = rows[0]
            val_cids.add(r_val["_cluster_id"])
            curr_val += r_val["size"]

            _, r_test = rows[1]
            test_cids.add(r_test["_cluster_id"])

            _, r_train = rows[2]
            train_cids.add(r_train["_cluster_id"])
            curr_train += r_train["size"]

            remaining_rows = rows[3:]
        else:
            remaining_rows = rows

        for _, row in remaining_rows:
            cid = row["_cluster_id"]
            cnt = row["size"]

            if curr_train + cnt <= target_train or (curr_train == 0 and ratios[0] > 0):
                train_cids.add(cid)
                curr_train += cnt
            elif curr_val + cnt <= target_val or (curr_val == 0 and ratios[1] > 0):
                val_cids.add(cid)
                curr_val += cnt
            else:
                test_cids.add(cid)


    train_df = valid_df[valid_df["_cluster_id"].isin(train_cids)].copy()
    val_df = valid_df[valid_df["_cluster_id"].isin(val_cids)].copy()
    test_df = valid_df[valid_df["_cluster_id"].isin(test_cids)].copy()

    # Add split tag
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    for df in [train_df, val_df, test_df]:
        df.drop(columns=["_cluster_id"], inplace=True)
        df.reset_index(drop=True, inplace=True)

    return train_df, val_df, test_df


def validate_severity_split_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    duplicate_clusters: list[list[str]] | None = None,
) -> dict[str, Any]:
    """Assert zero leakage across train, val, and test partitions.

    Raises:
        AssertionError: If any path, hash, or cluster is present in multiple splits.
    """
    # 1. Path overlap check
    train_paths = set(train_df["image_path"].astype(str))
    val_paths = set(val_df["image_path"].astype(str))
    test_paths = set(test_df["image_path"].astype(str))

    tv_paths = train_paths & val_paths
    tt_paths = train_paths & test_paths
    vt_paths = val_paths & test_paths

    assert not tv_paths, f"Path overlap in train and val: {len(tv_paths)}"
    assert not tt_paths, f"Path overlap in train and test: {len(tt_paths)}"
    assert not vt_paths, f"Path overlap in val and test: {len(vt_paths)}"

    # 2. Exact SHA-256 hash overlap check
    train_hashes = set(train_df["sha256"].astype(str))
    val_hashes = set(val_df["sha256"].astype(str))
    test_hashes = set(test_df["sha256"].astype(str))

    tv_hashes = train_hashes & val_hashes
    tt_hashes = train_hashes & test_hashes
    vt_hashes = val_hashes & test_hashes

    assert not tv_hashes, f"SHA-256 hash overlap in train and val: {len(tv_hashes)}"
    assert not tt_hashes, f"SHA-256 hash overlap in train and test: {len(tt_hashes)}"
    assert not vt_hashes, f"SHA-256 hash overlap in val and test: {len(vt_hashes)}"

    # 3. Duplicate cluster isolation check
    if duplicate_clusters:
        for idx, cluster in enumerate(duplicate_clusters):
            c_set = set(cluster)
            in_train = bool(c_set & train_paths)
            in_val = bool(c_set & val_paths)
            in_test = bool(c_set & test_paths)
            span_count = sum([in_train, in_val, in_test])
            assert (
                span_count <= 1
            ), f"Cluster {idx} spans {span_count} splits: train={in_train}, val={in_val}, test={in_test}"

    # 4. Ensure all 3 classes are present in every split
    for split_name, s_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        classes_present = set(s_df["label"].unique())
        missing = set(SEVERITY_CLASSES) - classes_present
        assert (
            not missing
        ), f"Split '{split_name}' is missing classes: {missing}. Present: {classes_present}"

    return {
        "status": "PASSED",
        "zero_leakage": True,
        "train_count": len(train_df),
        "val_count": len(val_df),
        "test_count": len(test_df),
        "total_count": len(train_df) + len(val_df) + len(test_df),
        "classes": sorted(list(train_df["label"].unique())),
    }


def save_severity_manifests(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    audit_report_df: pd.DataFrame | None = None,
    output_dir: str | Path = "data/manifests",
) -> dict[str, str]:
    """Save split DataFrames, class map, and summary to manifests directory.

    Returns:
        Dict mapping artifact keys to saved file paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest_cols = [
        "image_path",
        "filename",
        "label",
        "label_id",
        "sha256",
        "width",
        "height",
        "aspect_ratio",
        "blur_score",
        "brightness",
        "contrast",
        "split",
    ]

    def _prep_df(df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in manifest_cols if c in df.columns]
        return df[cols].copy()

    train_out = _prep_df(train_df)
    val_out = _prep_df(val_df)
    test_out = _prep_df(test_df)

    train_path = out / "severity_train.csv"
    val_path = out / "severity_val.csv"
    test_path = out / "severity_test.csv"
    class_map_path = out / "severity_class_map.json"
    summary_path = out / "severity_manifest_summary.json"

    train_out.to_csv(train_path, index=False)
    val_out.to_csv(val_path, index=False)
    test_out.to_csv(test_path, index=False)

    # Class map
    with open(class_map_path, "w") as f:
        json.dump(SEVERITY_ID_TO_CLASS, f, indent=2)

    # Manifest summary metadata
    total_samples = len(train_out) + len(val_out) + len(test_out)
    summary = {
        "dataset_name": "Car Damage Severity Dataset (3 classes)",
        "classes": SEVERITY_CLASSES,
        "class_to_id": SEVERITY_CLASS_TO_ID,
        "train_samples": len(train_out),
        "val_samples": len(val_out),
        "test_samples": len(test_out),
        "total_samples": total_samples,
        "train_pct": round(len(train_out) / total_samples * 100, 2),
        "val_pct": round(len(val_out) / total_samples * 100, 2),
        "test_pct": round(len(test_out) / total_samples * 100, 2),
        "class_distribution": {
            "train": train_out["label"].value_counts().to_dict(),
            "val": val_out["label"].value_counts().to_dict(),
            "test": test_out["label"].value_counts().to_dict(),
        },
        "files": {
            "train": train_path.name,
            "val": val_path.name,
            "test": test_path.name,
            "class_map": class_map_path.name,
        },
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    saved_files = {
        "train": str(train_path),
        "val": str(val_path),
        "test": str(test_path),
        "class_map": str(class_map_path),
        "summary": str(summary_path),
    }

    if audit_report_df is not None:
        report_path = out / "severity_audit_report.csv"
        audit_report_df.to_csv(report_path, index=False)
        saved_files["audit_report"] = str(report_path)

    return saved_files
