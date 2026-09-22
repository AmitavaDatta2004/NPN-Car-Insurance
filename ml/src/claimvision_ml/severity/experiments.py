"""Validation-only severity experiments for ML-IMPROVE-001.

Candidates are isolated from deployed artifacts. Test evaluation is a separate,
explicit operation after the validation selection has been reviewed and frozen.
"""

import hashlib
import json
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .cnn import SeverityCNN
from .inputs import read_rgb, resolve_image
from .mobilenet import SeverityMobileNetV2
from .vit import SeverityViTTiny

CLASSES = ["minor", "moderate", "severe"]


@dataclass(frozen=True)
class Recipe:
    architecture: str
    variant: str = "baseline"
    seed: int = 42
    epochs: int = 60
    warmup_epochs: int = 5
    patience: int = 12
    batch_size: int = 32
    workers: int = 2

    def __post_init__(self):
        if self.architecture not in {"cnn", "mobilenet", "vit", "vit_dual"}:
            raise ValueError("Unknown severity architecture")
        if self.variant not in {"baseline", "finetune", "full_image"}:
            raise ValueError("Unknown experiment variant")
        if min(self.epochs, self.patience, self.batch_size) < 1 or self.workers < 0:
            raise ValueError("Invalid training budget")
        if not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError("Warmup must be shorter than total training")


class BestCheckpoint:
    """Keep weights and score together, including across stage transitions."""

    def __init__(self):
        self.score = float("-inf")
        self.epoch = 0
        self.weights = None

    def update(self, model, score, epoch):
        if not np.isfinite(score):
            raise ValueError("Nonfinite validation score")
        if score > self.score:
            self.score, self.epoch = float(score), epoch
            self.weights = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            return True
        return False

    def restore(self, model):
        if self.weights is None:
            raise RuntimeError("No validated checkpoint exists")
        model.load_state_dict(self.weights)


class PadSquare:
    """Preserve the complete image; padding is an ablation, not a claimed gain."""

    def __call__(self, image):
        w, h = image.size
        side = max(w, h)
        left, top = (side - w) // 2, (side - h) // 2
        return ImageOps.expand(
            image, (left, top, side - w - left, side - h - top), fill=(124, 116, 104)
        )


def make_transform(size, train=False, full_image=False, mean=None, std=None):
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]
    if full_image:
        ops = [PadSquare(), transforms.Resize((size, size))]
    elif train:
        ops = [transforms.RandomResizedCrop(size, scale=(0.8, 1.0))]
    else:
        ops = [transforms.Resize(int(size * 256 / 224)), transforms.CenterCrop(size)]
    if train:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.15, 0.15, 0.1),
        ]
    return transforms.Compose(
        ops + [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_manifests(root):
    """Check frozen membership without inspecting test pixels or predictions."""
    root = Path(root)
    frames, checksums = {}, {}
    for split in ("train", "val", "test"):
        path = root / "data/manifests" / f"severity_{split}.csv"
        frame = pd.read_csv(path)
        if frame.empty or not {"image_path", "label", "label_id", "sha256"} <= set(
            frame
        ):
            raise ValueError(f"Invalid {split} manifest")
        if not all(
            row.label in CLASSES and CLASSES.index(row.label) == row.label_id
            for row in frame.itertuples()
        ):
            raise ValueError(f"Inconsistent label mapping in {split}")
        if set(frame.label_id) != {0, 1, 2}:
            raise ValueError(f"Missing severity class in {split}")
        frames[split], checksums[split] = frame, digest(path)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        for column in ("image_path", "sha256"):
            if set(frames[a][column]) & set(frames[b][column]):
                raise ValueError(f"Split leakage: {a}/{b} {column}")
    return frames, checksums


class ManifestImages(Dataset):
    def __init__(self, frame, root, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.paths = []
        for row in self.frame.itertuples():
            path = resolve_image(row.image_path, [root])
            if digest(path) != row.sha256:
                raise ValueError(
                    f"Image checksum differs from frozen manifest: {row.image_path}"
                )
            read_rgb(path)
            self.paths.append(path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return self.transform(read_rgb(self.paths[index])), int(
            self.frame.iloc[index].label_id
        )


def build_model(architecture, pretrained=True):
    if architecture == "cnn":
        return SeverityCNN()
    if architecture == "mobilenet":
        return SeverityMobileNetV2(pretrained=pretrained)
    if architecture == "vit":
        return SeverityViTTiny(pretrained=pretrained)
    if architecture == "vit_dual":
        from .dual_vit import DualStreamSeverityViT

        return DualStreamSeverityViT(pretrained=pretrained)
    raise ValueError("Unknown severity architecture")


def configure_stage(model, architecture, stage, full=False):
    for param in model.parameters():
        param.requires_grad = True
    if architecture == "cnn":
        return
    model.freeze_backbone()
    if stage == "finetune":
        if full:
            for param in model.parameters():
                param.requires_grad = True
        elif architecture == "mobilenet":
            model.unfreeze_final_blocks(2)
        else:
            model.unfreeze_top_blocks(4)


def train_mode(model):
    model.train()
    # requires_grad=False does not freeze BatchNorm running statistics.
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if not any(p.requires_grad for p in module.parameters()):
                module.eval()


def metrics_from_logits(logits, targets):
    probabilities = torch.softmax(torch.as_tensor(logits), 1).numpy()
    labels = np.asarray(targets)
    if not len(labels) or not np.isfinite(probabilities).all():
        raise ValueError("Empty or nonfinite model outputs")
    predicted = probabilities.argmax(1)
    confidence = probabilities.max(1)
    correct = predicted == labels
    ece = 0.0
    for low in np.arange(0, 1, 0.1):
        mask = (confidence > low) & (confidence <= low + 0.1)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(
            f1_score(
                labels, predicted, labels=[0, 1, 2], average="macro", zero_division=0
            )
        ),
        "weighted_f1": float(
            f1_score(labels, predicted, average="weighted", zero_division=0)
        ),
        "class_recall": recall_score(
            labels, predicted, labels=[0, 1, 2], average=None, zero_division=0
        ).tolist(),
        "confusion_matrix": confusion_matrix(
            labels, predicted, labels=[0, 1, 2]
        ).tolist(),
        "ece_10_bins": float(ece),
        "brier": float(
            np.mean(np.sum((probabilities - np.eye(3)[labels]) ** 2, axis=1))
        ),
        "ordinal_mae": float(np.abs(predicted - labels).mean()),
        "severe_as_minor": int(((labels == 2) & (predicted == 0)).sum()),
    }


def evaluate(model, loader, device):
    model.eval()
    outputs, labels = [], []
    with torch.inference_mode():
        for images, targets in loader:
            outputs.append(model(images.to(device)).cpu())
            labels.append(targets)
    logits, targets = torch.cat(outputs), torch.cat(labels)
    return metrics_from_logits(logits, targets), logits, targets


def run_experiment(root, recipe, run_id, pretrained=True):
    """Train one candidate. Never evaluate or select using the test set."""
    root = Path(root).resolve()
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a single directory name")
    frames, checksums = audit_manifests(root)
    random.seed(recipe.seed)
    np.random.seed(recipe.seed)
    torch.manual_seed(recipe.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(recipe.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(recipe.architecture, pretrained).to(device)
    size = 160 if recipe.architecture == "cnn" else 224
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if recipe.architecture in {"vit", "vit_dual"}:
        import timm

        config = timm.data.resolve_model_data_config(model.backbone)
        mean, std = config["mean"], config["std"]
    preprocessing = {
        "size": size,
        "full_image": recipe.variant == "full_image",
        "mean": list(mean),
        "std": list(std),
    }
    datasets = {
        split: ManifestImages(
            frames[split], root, make_transform(train=split == "train", **preprocessing)
        )
        for split in ("train", "val")
    }
    generator = torch.Generator().manual_seed(recipe.seed)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=recipe.batch_size,
            shuffle=split == "train",
            num_workers=recipe.workers,
            generator=generator,
            pin_memory=device == "cuda",
        )
        for split, dataset in datasets.items()
    }
    out = root / "artifacts/runs/improvements" / run_id
    out.mkdir(parents=True, exist_ok=False)
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    )
    record = {
        "task_id": "ML-IMPROVE-001",
        "run_id": run_id,
        "recipe": asdict(recipe),
        "manifest_sha256": checksums,
        "preprocessing": preprocessing,
        "git_commit": git.stdout.strip() or "unavailable",
        "device": device,
        "hardware": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "torch_version": str(torch.__version__),
        "software": {
            name: version(name)
            for name in ("torchvision", "timm", "numpy", "scikit-learn")
        },
        "started_utc": datetime.now(UTC).isoformat(),
        "training_protocol": {
            "loss": "CrossEntropyLoss (unweighted)",
            "optimizer": "AdamW",
            "weight_decay": 1e-4,
            "scheduler": "CosineAnnealingLR per stage",
            "eta_min": 1e-7,
            "gradient_clip_norm": 1.0,
            "head_warmup_lr": 1e-3,
            "head_finetune_lr": 1e-3 if recipe.architecture == "cnn" else 1e-4,
            "backbone_finetune_lr": 1e-3
            if recipe.architecture == "cnn"
            else (1e-5 if recipe.variant == "baseline" else 3e-5),
            "checkpoint_selection": "maximum validation macro F1 across both stages",
        },
        "pretrained_config": getattr(
            getattr(model, "backbone", None), "pretrained_cfg", None
        )
        if recipe.architecture.startswith("vit")
        else (
            "MobileNet_V2_Weights.IMAGENET1K_V1"
            if recipe.architecture == "mobilenet" and pretrained
            else None
        ),
        "pretrained": pretrained and recipe.architecture != "cnn",
        "status": "RUNNING",
        "test_evaluated": False,
    }
    (out / "experiment.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    start = time.perf_counter()
    best, history, stale, previous_stage = BestCheckpoint(), [], 0, None
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, recipe.epochs + 1):
        stage = (
            "warmup"
            if epoch <= recipe.warmup_epochs and recipe.architecture != "cnn"
            else "finetune"
        )
        if stage != previous_stage:
            if best.weights is not None:
                best.restore(model)
            configure_stage(
                model, recipe.architecture, stage, full=recipe.variant != "baseline"
            )
            head, backbone = [], []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    (
                        head
                        if "classifier" in name
                        or ".head." in name
                        or name.startswith("head.")
                        else backbone
                    ).append(param)
            backbone_lr = (
                1e-3
                if recipe.architecture == "cnn"
                else (1e-5 if recipe.variant == "baseline" else 3e-5)
            )
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": head,
                        "lr": 1e-3
                        if stage == "warmup"
                        else (1e-3 if recipe.architecture == "cnn" else 1e-4),
                    },
                    {"params": backbone, "lr": backbone_lr},
                ],
                weight_decay=1e-4,
            )
            stage_epochs = (
                recipe.warmup_epochs if stage == "warmup" else recipe.epochs - epoch + 1
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=stage_epochs, eta_min=1e-7
            )
            previous_stage, stale = stage, 0
        train_mode(model)
        loss_sum, count = 0.0, 0
        for images, targets in loaders["train"]:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(targets)
            count += len(targets)
        metrics, _, _ = evaluate(model, loaders["val"], device)
        scheduler.step()
        improved = best.update(model, metrics["macro_f1"], epoch)
        stale = 0 if improved else stale + 1
        if improved:
            torch.save(
                {
                    "state_dict": best.weights,
                    "architecture": recipe.architecture,
                    "epoch": best.epoch,
                    "preprocessing": preprocessing,
                    "classes": CLASSES,
                    "validation": metrics,
                    "recipe": asdict(recipe),
                },
                out / "best.pt",
            )
        history.append(
            {"epoch": epoch, "stage": stage, "train_loss": loss_sum / count, **metrics}
        )
        (out / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(
            f"{run_id} epoch {epoch}: loss={loss_sum / count:.4f}, val F1={metrics['macro_f1']:.4f}"
        )
        if stage == "finetune" and stale >= recipe.patience:
            break
    best.restore(model)
    metrics, logits, targets = evaluate(model, loaders["val"], device)
    if abs(metrics["macro_f1"] - best.score) > 1e-10:
        raise RuntimeError(
            "Restored checkpoint does not reproduce selected validation score"
        )
    np.savez(
        out / "validation_predictions.npz",
        logits=logits.numpy(),
        targets=targets.numpy(),
        image_paths=frames["val"].image_path.to_numpy(dtype=str),
    )
    loaded, _ = load_candidate(out / "best.pt")
    sample = next(iter(loaders["val"]))[0][:2]
    with torch.inference_mode():
        torch.testing.assert_close(
            loaded(sample), model(sample.to(device)).cpu(), rtol=1e-3, atol=1e-4
        )
    timings = []
    with torch.inference_mode():
        for index in range(13):
            started = time.perf_counter()
            loaded(sample[:1])
            if index >= 3:
                timings.append((time.perf_counter() - started) * 1000)
    record.update(
        status="COMPLETE_VALIDATION_ONLY",
        validation=metrics,
        best_epoch=best.epoch,
        duration_seconds=time.perf_counter() - start,
        checkpoint_sha256=digest(out / "best.pt"),
        model_size_mb=(out / "best.pt").stat().st_size / 2**20,
        cpu_forward_median_ms=float(np.median(timings)),
    )
    (out / "experiment.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def load_candidate(checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_model(payload["architecture"], pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, make_transform(**payload["preprocessing"])


def compare_runs(root, run_ids):
    records = [
        json.loads(
            (
                Path(root) / "artifacts/runs/improvements" / run / "experiment.json"
            ).read_text()
        )
        for run in run_ids
    ]
    if any(r["status"] != "COMPLETE_VALIDATION_ONLY" for r in records):
        raise ValueError("Only completed validation experiments can be compared")
    if len({json.dumps(r["manifest_sha256"], sort_keys=True) for r in records}) != 1:
        raise ValueError("Cannot compare experiments using different frozen splits")
    return pd.DataFrame(
        [
            {
                "run_id": r["run_id"],
                "architecture": r["recipe"]["architecture"],
                "variant": r["recipe"]["variant"],
                "seed": r["recipe"]["seed"],
                "val_macro_f1": r["validation"]["macro_f1"],
                "val_accuracy": r["validation"]["accuracy"],
                "minor_recall": r["validation"]["class_recall"][0],
                "moderate_recall": r["validation"]["class_recall"][1],
                "severe_recall": r["validation"]["class_recall"][2],
                "ece": r["validation"]["ece_10_bins"],
                "cpu_ms": r["cpu_forward_median_ms"],
            }
            for r in records
        ]
    ).sort_values("val_macro_f1", ascending=False)
