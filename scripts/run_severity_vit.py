"""Headless training and evaluation runner for ViT-Tiny severity classification (SEV-VIT-001).

Runs Stage A + Stage B training, early stopping on validation macro F1,
single test set evaluation, latency benchmarking, and artifacts export.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Add ml/src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "ml" / "src"))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from claimvision_ml.severity.vit import (
    CLASS_TO_ID,
    ID_TO_CLASS,
    SEVERITY_CLASSES,
    SeverityDataset,
    SeverityViTTiny,
    build_vit_model,
    export_vit_onnx,
    get_vit_transforms,
    predict_severity_vit,
    save_vit_checkpoint,
)

SEED = 42


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(SEED)
    print("=" * 70)
    print("ClaimVision AI — Phase 7: ViT-Tiny Severity Classifier (SEV-VIT-001)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution device: {device}")

    manifest_dir = PROJECT_ROOT / "data" / "manifests"
    train_csv = manifest_dir / "severity_train.csv"
    val_csv = manifest_dir / "severity_val.csv"
    test_csv = manifest_dir / "severity_test.csv"

    for p in (train_csv, val_csv, test_csv):
        if not p.is_file():
            raise FileNotFoundError(f"Required manifest not found: {p}")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    print(f"Loaded manifests: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Preprocessing and datasets
    train_tf = get_vit_transforms(split="train", image_size=224)
    eval_tf = get_vit_transforms(split="val", image_size=224)

    train_dataset = SeverityDataset(train_df, transform=train_tf)
    val_dataset = SeverityDataset(val_df, transform=eval_tf)
    test_dataset = SeverityDataset(test_df, transform=eval_tf)

    batch_size = 16 if torch.cuda.is_available() else 8
    num_workers = 0

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Build model (Stage A: frozen backbone)
    print("\n[Architecture] Building ViT-Tiny (vit_tiny_patch16_224)...")
    model = build_vit_model(pretrained=True, num_classes=3, freeze_backbone=True)
    model.to(device)

    trainable_a, total_params = model.count_parameters()
    print(f"Total parameters: {total_params:,} | Stage A Trainable parameters: {trainable_a:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
        "val_severe_recall": [],
    }

    best_val_macro_f1 = 0.0
    best_epoch = -1
    model_save_path = PROJECT_ROOT / "artifacts" / "models" / "severity_vit.pt"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    def evaluate(model_eval, loader):
        model_eval.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets in loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model_eval(images)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * len(targets)

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targets.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        acc = accuracy_score(all_targets, all_preds)
        macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        recalls = recall_score(all_targets, all_preds, average=None, zero_division=0)
        severe_rec = recalls[CLASS_TO_ID["severe"]] if len(recalls) > 2 else 0.0

        return avg_loss, acc, macro_f1, severe_rec, all_preds, all_targets

    # STAGE A: Train head only
    stage_a_epochs = 3
    print(f"\n[Stage A] Warmup training classification head ({stage_a_epochs} epochs)...")
    optimizer_a = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-2)

    for epoch in range(1, stage_a_epochs + 1):
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer_a.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_a.step()
            train_loss += loss.item() * len(targets)

        train_loss /= len(train_dataset)
        val_loss, val_acc, val_f1, val_sev, _, _ = evaluate(model, val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_macro_f1"].append(val_f1)
        history["val_severe_recall"].append(val_sev)

        print(f"  Stage A - Epoch {epoch}/{stage_a_epochs} — Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro F1: {val_f1:.4f} | Severe Recall: {val_sev:.4f}")

        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            best_epoch = epoch
            save_vit_checkpoint(model, model_save_path, epoch=epoch, metrics={"val_macro_f1": val_f1})

    # STAGE B: Unfreeze top 4 blocks + norm layer
    stage_b_epochs = 5
    print(f"\n[Stage B] Fine-tuning top transformer blocks ({stage_b_epochs} epochs)...")
    model.unfreeze_top_blocks(num_blocks=4)
    trainable_b, _ = model.count_parameters()
    print(f"Stage B Trainable parameters: {trainable_b:,} / {total_params:,}")

    optimizer_b = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5, weight_decay=1e-2)
    scheduler_b = CosineAnnealingLR(optimizer_b, T_max=stage_b_epochs, eta_min=1e-6)

    for epoch in range(1, stage_b_epochs + 1):
        global_epoch = stage_a_epochs + epoch
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer_b.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_b.step()
            train_loss += loss.item() * len(targets)

        scheduler_b.step()
        train_loss /= len(train_dataset)
        val_loss, val_acc, val_f1, val_sev, _, _ = evaluate(model, val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_macro_f1"].append(val_f1)
        history["val_severe_recall"].append(val_sev)

        print(f"  Stage B - Epoch {epoch}/{stage_b_epochs} (Total {global_epoch}) — Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro F1: {val_f1:.4f} | Severe Recall: {val_sev:.4f}")

        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            best_epoch = global_epoch
            save_vit_checkpoint(model, model_save_path, epoch=global_epoch, metrics={"val_macro_f1": val_f1})

    print(f"\n[Training Complete] Best Validation Macro F1: {best_val_macro_f1:.4f} at epoch {best_epoch}")

    # Load best checkpoint for test evaluation
    if model_save_path.is_file():
        print(f"Loading best checkpoint from: {model_save_path}")
        best_model = SeverityViTTiny(pretrained=False, num_classes=3)
        ckpt = torch.load(str(model_save_path), map_location=device, weights_only=False)
        best_model.load_state_dict(ckpt["model_state_dict"])
        best_model.to(device)
    else:
        best_model = model

    # HELD-OUT TEST EVALUATION (Evaluated strictly ONCE)
    print("\n[Held-out Test Evaluation] Evaluating strictly once on severity_test.csv...")
    test_loss, test_acc, test_macro_f1, test_sev_rec, test_preds, test_targets = evaluate(best_model, test_loader)
    test_weighted_f1 = f1_score(test_targets, test_preds, average="weighted", zero_division=0)
    test_macro_prec = precision_score(test_targets, test_preds, average="macro", zero_division=0)
    test_macro_rec = recall_score(test_targets, test_preds, average="macro", zero_division=0)

    cm = confusion_matrix(test_targets, test_preds)
    report_dict = classification_report(test_targets, test_preds, target_names=SEVERITY_CLASSES, output_dict=True)

    print("\n--- Test Set Metrics ---")
    print(f"Test Accuracy:         {test_acc * 100:.2f}%")
    print(f"Test Macro F1:         {test_macro_f1:.4f}")
    print(f"Test Weighted F1:      {test_weighted_f1:.4f}")
    print(f"Test Macro Precision:  {test_macro_prec:.4f}")
    print(f"Test Macro Recall:     {test_macro_rec:.4f}")
    print(f"Severe Class Recall:   {test_sev_rec:.4f}")
    print("\nConfusion Matrix:\n", cm)

    # Latency benchmarking
    print("\n[Latency Benchmark] Measuring CPU inference speed (50 samples)...")
    best_model.to("cpu")
    best_model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = best_model(dummy_input)

    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = best_model(dummy_input)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_latency_ms = float(np.mean(latencies))
    std_latency_ms = float(np.std(latencies))
    print(f"CPU Latency: {avg_latency_ms:.2f} ms/image (+/- {std_latency_ms:.2f} ms)")

    # Model size
    model_size_mb = model_save_path.stat().st_size / (1024 * 1024) if model_save_path.is_file() else 22.0
    print(f"Model Checkpoint Size: {model_size_mb:.2f} MB")

    # Export ONNX
    onnx_path = PROJECT_ROOT / "artifacts" / "models" / "severity_vit.onnx"
    try:
        print("\n[ONNX Export] Exporting to ONNX format...")
        export_vit_onnx(best_model, onnx_path, device="cpu", verify=True)
        print(f"ONNX model saved to: {onnx_path}")
    except Exception as e:
        print(f"ONNX export skipped/warning: {e}")

    # Generate Plots
    results_dir = PROJECT_ROOT / "ml" / "results" / "severity" / "vit"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Loss & Macro F1 Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs_range, history["train_loss"], label="Train Loss", marker="o", color="#2b5c8f")
    ax1.plot(epochs_range, history["val_loss"], label="Val Loss", marker="s", color="#d95f02")
    ax1.axvline(x=stage_a_epochs + 0.5, color="gray", linestyle="--", label="Stage B Start")
    ax1.set_title("ViT-Tiny Cross-Entropy Loss (Stage A + B)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, history["val_macro_f1"], label="Val Macro F1", marker="^", color="#1b9e77")
    ax2.plot(epochs_range, history["val_severe_recall"], label="Val Severe Recall", marker="d", color="#e7298a")
    ax2.axvline(x=stage_a_epochs + 0.5, color="gray", linestyle="--", label="Stage B Start")
    ax2.set_title("Validation Macro F1 & Severe Recall", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    loss_curve_path = results_dir / "vit_loss_curve.png"
    plt.savefig(loss_curve_path, dpi=200)
    plt.close()

    # Plot 2: Confusion Matrix Heatmap
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=SEVERITY_CLASSES,
        yticklabels=SEVERITY_CLASSES,
        cbar=False,
    )
    plt.title(f"ViT-Tiny Test Confusion Matrix\n(Accuracy: {test_acc*100:.1f}%, Macro F1: {test_macro_f1:.4f})", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Severity", fontsize=11)
    plt.ylabel("Ground Truth Severity", fontsize=11)
    plt.tight_layout()
    cm_path = results_dir / "vit_confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # Export Artifacts
    artifacts_dir = PROJECT_ROOT / "ml" / "artifacts" / "severity" / "vit"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "model_id": "SEV-VIT-001",
        "architecture": "vit_tiny_patch16_224",
        "parameters": {
            "total": total_params,
            "trainable_stage_a": trainable_a,
            "trainable_stage_b": trainable_b,
        },
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_val_macro_f1, 4),
        "test_metrics": {
            "accuracy": round(test_acc, 4),
            "macro_f1": round(test_macro_f1, 4),
            "weighted_f1": round(test_weighted_f1, 4),
            "macro_precision": round(test_macro_prec, 4),
            "macro_recall": round(test_macro_rec, 4),
            "severe_recall": round(test_sev_rec, 4),
            "confusion_matrix": cm.tolist(),
            "per_class": report_dict,
        },
        "operational_metrics": {
            "cpu_latency_ms": round(avg_latency_ms, 2),
            "cpu_latency_std": round(std_latency_ms, 2),
            "model_size_mb": round(model_size_mb, 2),
        },
    }

    metrics_path = artifacts_dir / "severity_vit_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    history_path = artifacts_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    class_map_path = artifacts_dir / "class_map.json"
    with open(class_map_path, "w", encoding="utf-8") as f:
        json.dump(ID_TO_CLASS, f, indent=2)

    config_path = artifacts_dir / "preprocessing_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "image_size": 224,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "interpolation": "bicubic",
            },
            f,
            indent=2,
        )

    print(f"\n[Saved Artifacts]")
    print(f"  - Metrics JSON:  {metrics_path}")
    print(f"  - History JSON:  {history_path}")
    print(f"  - Figures:       {results_dir}")
    print("\nPhase 7 (SEV-VIT-001) training and evaluation complete successfully!")


if __name__ == "__main__":
    main()
