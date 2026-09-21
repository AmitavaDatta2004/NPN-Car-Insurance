"""Standalone CLI runner for training MobileNetV2 Severity Classifier (SEV-MNV2-001).

Usage:
    python scripts/train_severity_mobilenet.py [--epochs-a 5] [--epochs-b 10] [--batch-size 32] [--device cuda|cpu]
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader

# Add ml/src to sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ml" / "src"))

from claimvision_ml.severity import (
    SEVERITY_CLASSES,
    SeverityDataset,
    build_severity_mobilenet,
    export_onnx,
    get_severity_class_weights,
    get_severity_transforms,
    predict_severity,
    save_severity_checkpoint,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * len(labels)

            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    mean_loss = total_loss / len(all_labels)
    acc = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    macro_rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)

    minor_mask = (all_labels == 0)
    minor_rec = float((all_preds[minor_mask] == 0).mean()) if minor_mask.sum() > 0 else 0.0

    mod_mask = (all_labels == 1)
    mod_rec = float((all_preds[mod_mask] == 1).mean()) if mod_mask.sum() > 0 else 0.0

    severe_mask = (all_labels == 2)
    severe_rec = float((all_preds[severe_mask] == 2).mean()) if severe_mask.sum() > 0 else 0.0

    return {
        "loss": mean_loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_recall": macro_rec,
        "macro_precision": macro_prec,
        "minor_recall": minor_rec,
        "moderate_recall": mod_rec,
        "severe_recall": severe_rec,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Severity MobileNetV2")
    parser.add_argument("--epochs-a", type=int, default=10, help="Stage A epochs (head only)")
    parser.add_argument("--epochs-b", type=int, default=15, help="Stage B epochs (fine-tuning)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr-a", type=float, default=1e-3, help="Stage A learning rate")
    parser.add_argument("--lr-b-backbone", type=float, default=2e-5, help="Stage B backbone learning rate")
    parser.add_argument("--lr-b-head", type=float, default=2e-4, help="Stage B head learning rate")
    parser.add_argument("--lr-b", type=float, default=None, help="Stage B single fallback learning rate")
    parser.add_argument("--unfreeze-blocks", type=int, default=4, help="Number of terminal feature blocks to unfreeze in Stage B")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="CrossEntropy label smoothing")
    parser.add_argument("--severe-weight-mult", type=float, default=1.15, help="Multiplier for severe class weight")
    parser.add_argument("--moderate-weight-mult", type=float, default=1.10, help="Multiplier for moderate class weight")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="", help="cuda or cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Paths
    train_csv = REPO_ROOT / "data" / "manifests" / "severity_train.csv"
    val_csv = REPO_ROOT / "data" / "manifests" / "severity_val.csv"
    test_csv = REPO_ROOT / "data" / "manifests" / "severity_test.csv"
    results_dir = REPO_ROOT / "ml" / "results" / "severity"
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir = REPO_ROOT / "artifacts" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    alt_models_dir = REPO_ROOT / "ml" / "artifacts" / "severity"
    alt_models_dir.mkdir(parents=True, exist_ok=True)

    # DataLoaders
    batch_size = args.batch_size if device.type == "cuda" else min(args.batch_size, 16)
    train_dataset = SeverityDataset(train_csv, transform=get_severity_transforms("train"))
    val_dataset = SeverityDataset(val_csv, transform=get_severity_transforms("val"))
    test_dataset = SeverityDataset(test_csv, transform=get_severity_transforms("val"))

    num_workers = 2 if os.name != "nt" else 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    logger.info(f"Loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # Loss with balanced weights, moderate/severe multipliers & label smoothing
    class_weights = get_severity_class_weights(train_csv).to(device)
    class_weights[1] *= args.moderate_weight_mult
    class_weights[2] *= args.severe_weight_mult
    class_weights = class_weights / class_weights.mean()
    logger.info(f"Class weights: minor={class_weights[0]:.4f}, moderate={class_weights[1]:.4f}, severe={class_weights[2]:.4f}")
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    # Model
    model = build_severity_mobilenet(pretrained=True, num_classes=3, dropout=0.3).to(device)
    model.freeze_backbone()

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "val_macro_f1": [], "val_severe_recall": [],
        "val_moderate_recall": [], "val_minor_recall": [],
    }

    best_val_f1 = 0.0
    best_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())

    ckpt_path = alt_models_dir / "severity_mnv2_v1.pt"
    local_pt = models_dir / "severity_mnv2.pt"

    # Stage A
    logger.info(f"Starting Stage A ({args.epochs_a} epochs, backbone frozen)...")
    optimizer_a = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr_a, weight_decay=1e-4)

    patience_a = 0
    for epoch in range(1, args.epochs_a + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer_a, device)
        val_m = evaluate_model(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_m["loss"])
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_m["accuracy"])
        history["val_macro_f1"].append(val_m["macro_f1"])
        history["val_severe_recall"].append(val_m["severe_recall"])
        history["val_moderate_recall"].append(val_m["moderate_recall"])
        history["val_minor_recall"].append(val_m["minor_recall"])

        is_best = val_m["macro_f1"] > best_val_f1
        if is_best:
            best_val_f1 = val_m["macro_f1"]
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            patience_a = 0
            save_severity_checkpoint(
                model=model,
                save_path=ckpt_path,
                epoch=best_epoch,
                metrics=val_m,
            )
            save_severity_checkpoint(
                model=model,
                save_path=local_pt,
                epoch=best_epoch,
                metrics=val_m,
            )
        else:
            patience_a += 1

        logger.info(
            f"Stage A Ep {epoch:2d}/{args.epochs_a:2d} ({elapsed:.1f}s) | "
            f"TrLoss: {tr_loss:.4f} TrAcc: {tr_acc:.1%} | "
            f"ValLoss: {val_m['loss']:.4f} ValAcc: {val_m['accuracy']:.1%} | "
            f"ValF1: {val_m['macro_f1']:.4f} | ModRec: {val_m['moderate_recall']:.1%} | SevRec: {val_m['severe_recall']:.1%}"
            f"{' [BEST]' if is_best else ''}"
        )

        if patience_a >= args.patience and epoch >= 4:
            logger.info(f"Stage A early stopping triggered after {epoch} epochs (best epoch: {best_epoch}).")
            break

    # Restore best Stage A weights before starting Stage B
    model.load_state_dict(best_weights)
    logger.info(f"Loaded best Stage A weights from epoch {best_epoch} (Val F1: {best_val_f1:.4f}) for Stage B")

    # Stage B: Unfreeze top blocks and apply differential learning rates
    n_blocks = args.unfreeze_blocks
    logger.info(f"Starting Stage B ({args.epochs_b} epochs, fine-tuning top {n_blocks} blocks)...")
    model.unfreeze_final_blocks(n_blocks=n_blocks)

    lr_backbone = args.lr_b if args.lr_b is not None else args.lr_b_backbone
    lr_head = args.lr_b if args.lr_b is not None else args.lr_b_head

    backbone_params = [p for p in model.features.parameters() if p.requires_grad]
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]

    optimizer_b = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=1e-4,
    )
    scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_b, T_max=args.epochs_b, eta_min=1e-6)

    patience_b = 0
    for epoch in range(1, args.epochs_b + 1):
        tot_epoch = args.epochs_a + epoch
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer_b, device)
        val_m = evaluate_model(model, val_loader, criterion, device)
        scheduler_b.step()
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_m["loss"])
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_m["accuracy"])
        history["val_macro_f1"].append(val_m["macro_f1"])
        history["val_severe_recall"].append(val_m["severe_recall"])
        history["val_moderate_recall"].append(val_m["moderate_recall"])
        history["val_minor_recall"].append(val_m["minor_recall"])

        is_best = val_m["macro_f1"] > best_val_f1
        if is_best:
            best_val_f1 = val_m["macro_f1"]
            best_epoch = tot_epoch
            best_weights = copy.deepcopy(model.state_dict())
            patience_b = 0
            save_severity_checkpoint(
                model=model,
                save_path=ckpt_path,
                epoch=best_epoch,
                metrics=val_m,
            )
            save_severity_checkpoint(
                model=model,
                save_path=local_pt,
                epoch=best_epoch,
                metrics=val_m,
            )
        else:
            patience_b += 1

        logger.info(
            f"Stage B Ep {tot_epoch:2d}/{args.epochs_a + args.epochs_b:2d} ({elapsed:.1f}s) | "
            f"TrLoss: {tr_loss:.4f} TrAcc: {tr_acc:.1%} | "
            f"ValLoss: {val_m['loss']:.4f} ValAcc: {val_m['accuracy']:.1%} | "
            f"ValF1: {val_m['macro_f1']:.4f} | ModRec: {val_m['moderate_recall']:.1%} | SevRec: {val_m['severe_recall']:.1%}"
            f"{' [BEST]' if is_best else ''}"
        )

        if patience_b >= args.patience and epoch >= 4:
            logger.info(f"Stage B early stopping triggered after {epoch} epochs (best epoch: {best_epoch}).")
            break

    # Restore best weights across all stages
    model.load_state_dict(best_weights)
    logger.info(f"Restored best weights from epoch {best_epoch} (Val Macro F1 = {best_val_f1:.4f})")

    # Evaluation on Validation and Held-out Test
    val_m = evaluate_model(model, val_loader, criterion, device)
    test_m = evaluate_model(model, test_loader, criterion, device)

    logger.info("=== Final Results ===")
    logger.info(f"Validation: Acc={val_m['accuracy']:.4f}, MacroF1={val_m['macro_f1']:.4f}, SevereRecall={val_m['severe_recall']:.4f}")
    logger.info(f"Test:       Acc={test_m['accuracy']:.4f}, MacroF1={test_m['macro_f1']:.4f}, SevereRecall={test_m['severe_recall']:.4f}")

    # Latency benchmark on CPU
    model_cpu = build_severity_mobilenet(pretrained=False, num_classes=3)
    model_cpu.load_state_dict(best_weights)
    model_cpu.to("cpu")
    model_cpu.eval()
    dummy = torch.randn(1, 3, 224, 224)

    for _ in range(10):
        with torch.no_grad():
            _ = model_cpu(dummy)

    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model_cpu(dummy)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_lat = float(np.mean(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    fps = 1000.0 / mean_lat
    logger.info(f"CPU Latency: mean={mean_lat:.2f} ms, p95={p95_lat:.2f} ms, FPS={fps:.1f}")

    # Export ONNX
    onnx_path = models_dir / "severity_mnv2.onnx"
    try:
        export_onnx(model, onnx_path, verify=False)
        logger.info(f"ONNX exported to {onnx_path}")
    except (RuntimeError, ValueError, ImportError) as e:
        logger.warning(f"ONNX export notice: {e}")

    # Generate Learning Curves Plot
    epochs_range = list(range(1, len(history["train_loss"]) + 1))
    _fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    ax1.plot(epochs_range, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    ax1.plot(epochs_range, history["val_loss"], "r-s", label="Val Loss", markersize=4)
    ax1.axvline(x=args.epochs_a + 0.5, color="gray", linestyle="--", label="Stage B Transition")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss (CrossEntropy)", fontsize=12)
    ax1.set_title("Loss Curves", fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, [acc * 100 for acc in history["train_acc"]], "b-o", label="Train Acc", markersize=4)
    ax2.plot(epochs_range, [acc * 100 for acc in history["val_acc"]], "r-s", label="Val Acc", markersize=4)
    ax2.axvline(x=args.epochs_a + 0.5, color="gray", linestyle="--", label="Stage B Transition")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Accuracy Curves", fontsize=13, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.plot(epochs_range, history["val_macro_f1"], "g-^", label="Val Macro F1", markersize=5)
    ax3.plot(epochs_range, history["val_severe_recall"], "m-d", label="Val Severe Recall", markersize=5)
    ax3.axvline(x=args.epochs_a + 0.5, color="gray", linestyle="--", label="Stage B Transition")
    ax3.axvline(x=best_epoch, color="gold", linestyle=":", linewidth=2, label=f"Best Ep ({best_epoch})")
    ax3.set_xlabel("Epoch", fontsize=12)
    ax3.set_ylabel("Score (0.0 to 1.0)", fontsize=12)
    ax3.set_title("Validation Performance", fontsize=13, fontweight="bold")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.suptitle("Figure 2: MobileNetV2 Severity Classifier Training Progression", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / "mnv2_learning_curves.png", dpi=150)
    plt.close()

    # Generate Confusion Matrix Plots
    cm_val = confusion_matrix(val_m["labels"], val_m["predictions"])
    cm_val_norm = cm_val.astype("float") / cm_val.sum(axis=1)[:, np.newaxis]
    _fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm_val, annot=True, fmt="d", cmap="Blues", xticklabels=SEVERITY_CLASSES, yticklabels=SEVERITY_CLASSES, ax=ax1)
    ax1.set_title("Validation Confusion Matrix (Counts)", fontweight="bold")
    ax1.set_xlabel("Predicted Class")
    ax1.set_ylabel("True Class")
    sns.heatmap(cm_val_norm, annot=True, fmt=".1%", cmap="Greens", xticklabels=SEVERITY_CLASSES, yticklabels=SEVERITY_CLASSES, ax=ax2)
    ax2.set_title("Validation Confusion Matrix (Normalized)", fontweight="bold")
    ax2.set_xlabel("Predicted Class")
    ax2.set_ylabel("True Class")
    plt.suptitle("Figure 3: MobileNetV2 Validation Set Confusion Matrices", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / "mnv2_val_confusion_matrix.png", dpi=150)
    plt.close()

    cm_test = confusion_matrix(test_m["labels"], test_m["predictions"])
    cm_test_norm = cm_test.astype("float") / cm_test.sum(axis=1)[:, np.newaxis]
    _fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Oranges", xticklabels=SEVERITY_CLASSES, yticklabels=SEVERITY_CLASSES, ax=ax1)
    ax1.set_title("Test Confusion Matrix (Counts)", fontweight="bold")
    ax1.set_xlabel("Predicted Class")
    ax1.set_ylabel("True Class")
    sns.heatmap(cm_test_norm, annot=True, fmt=".1%", cmap="Reds", xticklabels=SEVERITY_CLASSES, yticklabels=SEVERITY_CLASSES, ax=ax2)
    ax2.set_title("Test Confusion Matrix (Normalized)", fontweight="bold")
    ax2.set_xlabel("Predicted Class")
    ax2.set_ylabel("True Class")
    plt.suptitle("Figure 5: MobileNetV2 Held-out Test Confusion Matrices", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / "mnv2_test_confusion_matrix.png", dpi=150)
    plt.close()

    # Save Metrics Summary JSON
    metrics_summary = {
        "experiment_id": "SEV-MNV2-001",
        "model_name": "MobileNetV2",
        "training_type": "two-stage transfer learning",
        "seed": args.seed,
        "stage_a_epochs": args.epochs_a,
        "stage_b_epochs": args.epochs_b,
        "best_epoch": int(best_epoch),
        "validation": {
            "loss": round(float(val_m["loss"]), 4),
            "accuracy": round(float(val_m["accuracy"]), 4),
            "macro_precision": round(float(val_m["macro_precision"]), 4),
            "macro_recall": round(float(val_m["macro_recall"]), 4),
            "macro_f1": round(float(val_m["macro_f1"]), 4),
            "minor_recall": round(float(val_m["minor_recall"]), 4),
            "moderate_recall": round(float(val_m["moderate_recall"]), 4),
            "severe_recall": round(float(val_m["severe_recall"]), 4),
        },
        "test": {
            "loss": round(float(test_m["loss"]), 4),
            "accuracy": round(float(test_m["accuracy"]), 4),
            "macro_precision": round(float(test_m["macro_precision"]), 4),
            "macro_recall": round(float(test_m["macro_recall"]), 4),
            "macro_f1": round(float(test_m["macro_f1"]), 4),
            "weighted_f1": round(float(test_m["weighted_f1"]), 4),
            "minor_recall": round(float(test_m["minor_recall"]), 4),
            "moderate_recall": round(float(test_m["moderate_recall"]), 4),
            "severe_recall": round(float(test_m["severe_recall"]), 4),
            "confusion_matrix": cm_test.tolist(),
        },
        "latency": {
            "cpu_mean_ms": round(mean_lat, 2),
            "cpu_p95_ms": round(p95_lat, 2),
            "fps": round(fps, 1),
        },
        "parameters": {
            "total": sum(p.numel() for p in model.parameters()),
            "trainable_stage_b": sum(p.numel() for p in model.parameters() if p.requires_grad),
        },
        "artifacts": {
            "checkpoint": str(local_pt.relative_to(REPO_ROOT)),
            "onnx": str(onnx_path.relative_to(REPO_ROOT)),
        }
    }

    metrics_json_path = REPO_ROOT / "ml" / "results" / "severity_mnv2_metrics.json"
    alt_metrics_json_path = REPO_ROOT / "ml" / "results" / "severity" / "mnv2_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    with open(alt_metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info(f"Summary metrics exported to {metrics_json_path} and {alt_metrics_json_path}")

    # Standalone smoke test
    sample_img = REPO_ROOT / "data" / "raw" / "car_damage_severity" / "data3a" / "training" / "01-minor" / "0001.JPEG"
    if sample_img.is_file():
        res = predict_severity(sample_img, model, device="cpu")
        logger.info(f"Standalone smoke test result: class={res.predicted_class}, conf={res.confidence:.2%}, time={res.inference_time_ms:.2f}ms")

    logger.info("Training script execution finished successfully.")


if __name__ == "__main__":
    main()
