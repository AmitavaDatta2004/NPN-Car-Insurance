"""Group-aware, duplicate-safe dataset splitting and manifest management.

Enforces zero split leakage across train, validation, and test partitions
for vehicle claims, duplicate image groups, and visual evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def create_group_aware_split(
    df: pd.DataFrame,
    group_col: str = "claim_id",
    duplicate_clusters: list[list[str]] | None = None,
    filepath_col: str = "path",
    label_col: str = "label",
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a stratified, group-aware, duplicate-safe train/val/test split.

    Args:
        df: Input DataFrame containing image records.
        group_col: Column representing claim or entity group.
        duplicate_clusters: Optional list of near/exact duplicate image clusters.
        filepath_col: Column matching paths in duplicate_clusters.
        label_col: Class label column for stratified distribution.
        ratios: Proportions for (train, val, test).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    if abs(sum(ratios) - 1.0) > 1e-4:
        raise ValueError(f"Ratios must sum to 1.0, got: {ratios}")

    df_copy = df.copy()

    # Create a unified grouping key
    # If no group_col exists in DataFrame, fallback to individual image indexing
    if group_col not in df_copy.columns:
        df_copy[group_col] = [f"group_{i}" for i in range(len(df_copy))]

    # Map duplicate clusters into a merged meta-group
    path_to_group = dict(
        zip(df_copy[filepath_col].astype(str), df_copy[group_col].astype(str))
    )
    group_mapping = {g: g for g in df_copy[group_col].astype(str).unique()}

    if duplicate_clusters:
        for cluster in duplicate_clusters:
            # Find groups present in this cluster
            cluster_groups = {
                path_to_group[str(p)] for p in cluster if str(p) in path_to_group
            }
            if len(cluster_groups) > 1:
                # Merge all to one canonical group id
                canonical = sorted(cluster_groups)[0]
                for g in cluster_groups:
                    group_mapping[g] = canonical

    df_copy["_split_group"] = df_copy[group_col].astype(str).map(group_mapping)

    # Determine dominant label per split group for stratified assignment
    group_summaries = []
    rng = np.random.RandomState(seed)

    for grp_id, grp_df in df_copy.groupby("_split_group"):
        dominant_label = (
            grp_df[label_col].mode().iloc[0] if label_col in grp_df else "default"
        )
        group_summaries.append(
            {
                "_split_group": grp_id,
                "count": len(grp_df),
                "dominant_label": str(dominant_label),
            }
        )

    group_df = pd.DataFrame(group_summaries)

    train_groups, val_groups, test_groups = set(), set(), set()

    # Stratified partition by dominant label
    for _, class_group in group_df.groupby("dominant_label"):
        shuffled_groups = class_group.sample(frac=1.0, random_state=rng)
        total_items = class_group["count"].sum()
        target_train = total_items * ratios[0]
        target_val = total_items * ratios[1]

        curr_train, curr_val = 0, 0
        for _, row in shuffled_groups.iterrows():
            grp = row["_split_group"]
            cnt = row["count"]

            if curr_train + cnt <= target_train or (curr_train == 0 and ratios[0] > 0):
                train_groups.add(grp)
                curr_train += cnt
            elif curr_val + cnt <= target_val or (curr_val == 0 and ratios[1] > 0):
                val_groups.add(grp)
                curr_val += cnt
            else:
                test_groups.add(grp)

    # Filter original dataframe into splits
    train_df = df_copy[df_copy["_split_group"].isin(train_groups)].drop(
        columns=["_split_group"]
    )
    val_df = df_copy[df_copy["_split_group"].isin(val_groups)].drop(
        columns=["_split_group"]
    )
    test_df = df_copy[df_copy["_split_group"].isin(test_groups)].drop(
        columns=["_split_group"]
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def validate_split_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_col: str = "claim_id",
    hash_col: str | None = "sha256",
    duplicate_clusters: list[list[str]] | None = None,
    filepath_col: str = "path",
) -> dict[str, Any]:
    """Verify zero overlap of claim IDs, file hashes, and duplicate clusters between splits.

    Raises:
        AssertionError: If any intersection is found between train, val, or test.

    Returns:
        Summary dictionary with split counts and confirmation of zero leakage.
    """
    # Check 1: Group column overlap
    if (
        group_col in train_df.columns
        and group_col in val_df.columns
        and group_col in test_df.columns
    ):
        train_groups = set(train_df[group_col].dropna().astype(str))
        val_groups = set(val_df[group_col].dropna().astype(str))
        test_groups = set(test_df[group_col].dropna().astype(str))

        tv = train_groups & val_groups
        tt = train_groups & test_groups
        vt = val_groups & test_groups

        assert not tv, (
            f"Split leakage detected: {len(tv)} claim groups in both train and val!"
        )
        assert not tt, (
            f"Split leakage detected: {len(tt)} claim groups in both train and test!"
        )
        assert not vt, (
            f"Split leakage detected: {len(vt)} claim groups in both val and test!"
        )

    # Check 2: Exact file hash overlap
    if (
        hash_col
        and hash_col in train_df.columns
        and hash_col in val_df.columns
        and hash_col in test_df.columns
    ):
        train_hashes = set(train_df[hash_col].dropna().astype(str))
        val_hashes = set(val_df[hash_col].dropna().astype(str))
        test_hashes = set(test_df[hash_col].dropna().astype(str))

        hv = train_hashes & val_hashes
        ht = train_hashes & test_hashes
        hvt = val_hashes & test_hashes

        assert not hv, (
            f"Exact duplicate leakage detected: {len(hv)} hashes in train and val!"
        )
        assert not ht, (
            f"Exact duplicate leakage detected: {len(ht)} hashes in train and test!"
        )
        assert not hvt, (
            f"Exact duplicate leakage detected: {len(hvt)} hashes in val and test!"
        )

    # Check 3: Duplicate clusters across splits
    if duplicate_clusters and filepath_col in train_df.columns:
        train_paths = set(train_df[filepath_col].astype(str))
        val_paths = set(val_df[filepath_col].astype(str))
        test_paths = set(test_df[filepath_col].astype(str))

        for idx, cluster in enumerate(duplicate_clusters):
            c_set = set(str(p) for p in cluster)
            in_train = bool(c_set & train_paths)
            in_val = bool(c_set & val_paths)
            in_test = bool(c_set & test_paths)

            leak_count = sum([in_train, in_val, in_test])
            assert leak_count <= 1, (
                f"Perceptual duplicate cluster #{idx} spans multiple splits! "
                f"(train={in_train}, val={in_val}, test={in_test})"
            )

    return {
        "status": "PASSED",
        "zero_leakage_verified": True,
        "train_count": len(train_df),
        "val_count": len(val_df),
        "test_count": len(test_df),
        "total": len(train_df) + len(val_df) + len(test_df),
    }


def save_manifests(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str | Path = "data/manifests",
    prefix: str = "fraud",
) -> dict[str, str]:
    """Save split DataFrames to CSV manifests and record split summary metadata."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_path = out / f"{prefix}_train.csv"
    val_path = out / f"{prefix}_val.csv"
    test_path = out / f"{prefix}_test.csv"
    meta_path = out / f"{prefix}_manifest_summary.json"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    summary = {
        "manifest_prefix": prefix,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "total_samples": len(train_df) + len(val_df) + len(test_df),
        "files": {
            "train": str(train_path.name),
            "val": str(val_path.name),
            "test": str(test_path.name),
        },
    }

    with open(meta_path, "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "train": str(train_path),
        "val": str(val_path),
        "test": str(test_path),
        "summary": str(meta_path),
    }
