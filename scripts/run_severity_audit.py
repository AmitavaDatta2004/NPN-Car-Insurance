"""Run full severity dataset audit, generate plots, and export frozen manifests."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from claimvision_ml.data.severity_audit import (
    SEVERITY_CLASSES,
    audit_severity_images,
    create_severity_split,
    discover_severity_images,
    find_exact_duplicate_groups,
    find_perceptual_duplicates,
    save_severity_manifests,
    validate_severity_split_leakage,
)

# Set styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10


def main():
    print("=" * 70)
    print("ClaimVision AI — Phase 4: Severity Dataset Audit & Manifest Freeze")
    print("=" * 70)

    raw_dir = Path("data/raw/car_damage_severity")
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw severity dataset directory missing at {raw_dir}")

    # 1. Discover all images
    print("\n[Step 1/6] Discovering severity images...")
    df = discover_severity_images(raw_dir)
    print(f"  Found {len(df)} total images across {df['original_split'].nunique()} original splits.")
    print("  Class counts in raw dataset:")
    for cls_name, count in df["label"].value_counts().items():
        print(f"    - {cls_name:8s}: {count} images ({count / len(df) * 100:.1f}%)")

    # 2. Audit images
    print("\n[Step 2/6] Auditing images (OpenCV decode, dimensions, blur, brightness, contrast, SHA-256, pHash)...")
    audit_df = audit_severity_images(df)
    corrupt_count = audit_df["is_corrupt"].sum()
    print(f"  Corrupt/unreadable images: {corrupt_count}")
    if corrupt_count > 0:
        print("  WARNING: Quarantining corrupt files from downstream manifests.")

    # 3. Detect exact and near duplicates
    print("\n[Step 3/6] Detecting exact duplicates (SHA-256) and near-duplicates (pHash <= 8)...")
    exact_dupes = find_exact_duplicate_groups(audit_df)
    print(f"  Exact duplicate groups (SHA-256): {len(exact_dupes)}")
    for h, paths in list(exact_dupes.items())[:3]:
        print(f"    Hash {h[:12]}...: {[Path(p).name for p in paths]}")

    near_clusters = find_perceptual_duplicates(audit_df, threshold=8)
    print(f"  Perceptual near-duplicate clusters: {len(near_clusters)}")
    total_near = sum(len(c) for c in near_clusters)
    print(f"  Total images in near-duplicate clusters: {total_near}")

    # 4. Generate frozen stratified 70/15/15 split
    print("\n[Step 4/6] Creating duplicate-safe 70/15/15 stratified train/val/test splits...")
    train_df, val_df, test_df = create_severity_split(
        audit_df,
        duplicate_clusters=near_clusters,
        ratios=(0.70, 0.15, 0.15),
        seed=42,
    )
    total_valid = len(train_df) + len(val_df) + len(test_df)
    print(f"  Train: {len(train_df):4d} ({len(train_df) / total_valid * 100:.1f}%)")
    print(f"  Val  : {len(val_df):4d} ({len(val_df) / total_valid * 100:.1f}%)")
    print(f"  Test : {len(test_df):4d} ({len(test_df) / total_valid * 100:.1f}%)")
    print(f"  Total: {total_valid:4d}")

    # 5. Validate zero-leakage assertions
    print("\n[Step 5/6] Validating anti-leakage invariants across splits...")
    leakage_res = validate_severity_split_leakage(train_df, val_df, test_df, duplicate_clusters=near_clusters)
    print(f"  Zero leakage verification: {leakage_res['status']} (zero_leakage={leakage_res['zero_leakage']})")
    print(f"  Classes present in every split: {leakage_res['classes']}")

    # 6. Save manifests
    print("\n[Step 6/6] Saving frozen manifests and generating audit plots...")
    manifest_dir = Path("data/manifests")
    saved_manifests = save_severity_manifests(
        train_df, val_df, test_df, audit_report_df=audit_df, output_dir=manifest_dir
    )
    for k, p in saved_manifests.items():
        print(f"  Saved {k:12s} -> {p}")

    # Generate Visualizations
    results_dir = Path("ml/results/severity")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Class Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    class_counts = audit_df[~audit_df["is_corrupt"]]["label"].value_counts().reindex(SEVERITY_CLASSES)
    bars = ax.bar(class_counts.index.str.capitalize(), class_counts.values, color=["#4CAF50", "#FF9800", "#F44336"], width=0.55, edgecolor="black", linewidth=1.2)
    ax.set_title("Car Damage Severity Dataset — Class Distribution (N = 1,631)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Images", fontsize=11)
    ax.set_xlabel("Damage Severity Class", fontsize=11)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height}\n({height / len(audit_df) * 100:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(class_counts.values) * 1.18)
    plt.tight_layout()
    chart1_path = results_dir / "class_distribution.png"
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"  Chart saved: {chart1_path}")

    # Plot 2: Split Proportions per Class
    split_summary = []
    for split_name, s_df in [("Train (70%)", train_df), ("Val (15%)", val_df), ("Test (15%)", test_df)]:
        vc = s_df["label"].value_counts().reindex(SEVERITY_CLASSES)
        for cls_name, cnt in vc.items():
            split_summary.append({"Split": split_name, "Severity": cls_name.capitalize(), "Count": cnt})
    split_sum_df = pd.DataFrame(split_summary)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=split_sum_df, x="Split", y="Count", hue="Severity", palette=["#4CAF50", "#FF9800", "#F44336"], ax=ax, edgecolor="black")
    ax.set_title("Duplicate-Safe Stratified Split Distribution Across Partitions", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Image Count", fontsize=11)
    ax.set_xlabel("Split Partition", fontsize=11)
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{int(h)}", (p.get_x() + p.get_width() / 2., h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    ax.legend(title="Severity", loc="upper right")
    plt.tight_layout()
    chart2_path = results_dir / "split_distribution.png"
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"  Chart saved: {chart2_path}")

    # Plot 3: Quality Metrics Distribution (Width, Height, Blur, Brightness, Contrast)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics_to_plot = [
        ("blur_score", "Laplacian Blur Variance", axes[0, 0]),
        ("brightness", "Mean Grayscale Brightness", axes[0, 1]),
        ("contrast", "Grayscale Contrast (Std Dev)", axes[1, 0]),
        ("aspect_ratio", "Aspect Ratio (Width / Height)", axes[1, 1]),
    ]
    palette = {"minor": "#4CAF50", "moderate": "#FF9800", "severe": "#F44336"}

    for col, title, ax in metrics_to_plot:
        sns.boxplot(data=audit_df[~audit_df["is_corrupt"]], x="label", y=col, order=SEVERITY_CLASSES, palette=palette, ax=ax, width=0.5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Severity Class")
        ax.set_ylabel("Value")

    fig.suptitle("Image Quality and Capture Distributions Across Severity Classes", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    chart3_path = results_dir / "quality_distributions.png"
    plt.savefig(chart3_path, dpi=300)
    plt.close()
    print(f"  Chart saved: {chart3_path}")

    # Plot 4: Sample Image Grid (3 classes x 4 representative images)
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    valid_df = audit_df[~audit_df["is_corrupt"]]

    for row_idx, cls_name in enumerate(SEVERITY_CLASSES):
        cls_subset = valid_df[valid_df["label"] == cls_name].sort_values("blur_score", ascending=False)
        # pick 4 diverse samples
        step = max(len(cls_subset) // 4, 1)
        sample_indices = [0, step, step * 2, min(step * 3, len(cls_subset) - 1)]

        for col_idx, s_idx in enumerate(sample_indices):
            row = cls_subset.iloc[s_idx]
            img_path = Path(row["image_path"])
            ax = axes[row_idx, col_idx]

            try:
                img_rgb = Image.open(img_path).convert("RGB")
                ax.imshow(img_rgb)
            except Exception:
                ax.text(0.5, 0.5, "Image Error", ha="center", va="center")

            ax.set_title(f"{cls_name.capitalize()} #{col_idx+1}\nBlur: {row['blur_score']:.1f} | Dim: {row['width']}x{row['height']}", fontsize=9)
            ax.axis("off")

    fig.suptitle("Representative Vehicle Damage Photographs by Severity Class", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    chart4_path = results_dir / "sample_grid.png"
    plt.savefig(chart4_path, dpi=300)
    plt.close()
    print(f"  Chart saved: {chart4_path}")

    print("\n[SUCCESS] Severity Dataset Audit complete! All manifests and charts generated successfully.")



if __name__ == "__main__":
    main()
