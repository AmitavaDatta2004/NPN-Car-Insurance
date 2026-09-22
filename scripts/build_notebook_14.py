#!/usr/bin/env python3
"""build_notebook_14.py — Generate notebooks/14_location_efficientnet_training.ipynb.

Task ID : LOC-EFF-001
Run     : python scripts/build_notebook_14.py
Output  : notebooks/14_location_efficientnet_training.ipynb
"""

from __future__ import annotations
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_PATH = REPO_ROOT / "notebooks" / "14_location_efficientnet_training.ipynb"

cells = []

def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source, "outputs": [], "execution_count": None}

def code(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": source, "outputs": [], "execution_count": None}

# ── Cell 0 ──────────────────────────────────────────────────────────────────
cells.append(md("""# Notebook 14 — Location Classification: EfficientNet-B0 Training & Evaluation

**Task ID:** LOC-EFF-001  
**Phase:** 11b — Damaged-Part Location CNN (EfficientNet-B0)  
**Owner:** Detection ML Member / Antigravity  
**Dataset:** COCO Car Damage Detection Dataset (59 train / 11 val / 8 test images)  
**Date:** 2026-09-22  
**Environment:** Google Colab (GPU recommended)

## Purpose

Train an **EfficientNet-B0 image-level classifier** to predict which vehicle part is visibly damaged.  
This notebook uses the **same dataset, same label derivation, and same evaluation protocol** as Notebook 13  
(MobileNetV2) so that results are directly comparable in Notebook 15.

| Class ID | Part Name |
|---|---|
| 0 | headlamp |
| 1 | front_bumper |
| 2 | hood |
| 3 | door |
| 4 | rear_bumper |

## Hypothesis

EfficientNet-B0, with its compound scaling and superior feature extraction (≥MobileNetV2 on ImageNet),  
will achieve higher macro F1 than MobileNetV2 on this 5-class location task.

> **Dataset Handling Note:**
> Identical to Notebook 13, `load_location_splits()` pools available annotated images and constructs a
> stratified 80/20 train/val split (48 train / 12 val) ensuring every class is represented in validation.
"""))

# ── Cell 1 — Environment ────────────────────────────────────────────────────
cells.append(code("""# Cell 1 — Environment, seeds, versions (identical to Notebook 13 for reproducibility)
import os, sys, random, importlib
from pathlib import Path
import numpy as np
import torch

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

IN_COLAB = 'google.colab' in sys.modules
try:
    import google.colab; IN_COLAB = True
except ImportError:
    pass

if IN_COLAB:
    COLAB_BASE = Path('/content/NPN-Car-Insurance')
    if not COLAB_BASE.exists():
        import subprocess
        subprocess.run(['git', 'clone', 'https://github.com/AmitavaDatta2004/NPN-Car-Insurance.git', str(COLAB_BASE)], check=True)
    REPO_ROOT = COLAB_BASE
else:
    REPO_ROOT = Path.cwd()
    while not (REPO_ROOT / 'ml').exists() and REPO_ROOT != REPO_ROOT.parent:
        REPO_ROOT = REPO_ROOT.parent

ML_SRC = REPO_ROOT / 'ml' / 'src'
if str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device} | Seed: {SEED} | Repo: {REPO_ROOT}')
for pkg in ['torch', 'torchvision', 'timm', 'numpy']:
    try:
        mod = importlib.import_module(pkg)
        print(f'  {pkg:<15} {getattr(mod, \"__version__\", \"unknown\")}')
    except ImportError:
        print(f'  {pkg:<15} NOT INSTALLED')
"""))

# ── Cell 2 — Dataset Audit ──────────────────────────────────────────────────
cells.append(code("""# Cell 2 — Locate raw COCO dataset and audit physical files on disk
from pathlib import Path
from claimvision_ml.location.dataset import LOCATION_CLASSES

RAW_CANDIDATES = [
    REPO_ROOT / 'data' / 'raw' / 'coco_car_damage' / 'coco-car-damage-detection-dataset' / 'coco-car-damage-detection-dataset',
    REPO_ROOT / 'data' / 'raw' / 'coco_car_damage' / 'coco-car-damage-detection-dataset',
    REPO_ROOT / 'data' / 'raw' / 'coco_car_damage',
    Path('/content/NPN-Car-Insurance/data/raw/coco_car_damage/coco-car-damage-detection-dataset/coco-car-damage-detection-dataset'),
    Path('/content/data/raw/coco_car_damage/coco-car-damage-detection-dataset/coco-car-damage-detection-dataset'),
]
RAW_DIR = None
for c in RAW_CANDIDATES:
    if (c / 'train').exists() and (c / 'val').exists():
        RAW_DIR = c; break
if RAW_DIR is None:
    raise RuntimeError('COCO raw dataset not found. Run Notebook 00 first.')
print(f'RAW_DIR: {RAW_DIR}\\n')

print('=== Raw Physical Disk Audit ===')
for s in ['train', 'val', 'test']:
    sdir = RAW_DIR / s
    if sdir.exists():
        imgs = [f for f in sdir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        annos = list(sdir.glob('*.json'))
        print(f'  [{s:<5}] {len(imgs):>2} physical image files on disk, {len(annos)} JSON annotation files')
"""))

# ── Cell 3 — Sample grid ────────────────────────────────────────────────────
cells.append(code("""# Cell 3 — Sample image grid per class (same images as NB 13 for direct comparison)
import cv2, matplotlib.pyplot as plt
from claimvision_ml.location.dataset import derive_location_labels

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
shown = {cls: False for cls in LOCATION_CLASSES}
train_anno_path = RAW_DIR / 'train' / 'COCO_mul_train_annos.json'
train_lmap = derive_location_labels(train_anno_path) if train_anno_path.exists() else {}

for fname, (label_id, label_name) in train_lmap.items():
    if shown[label_name]: continue
    img = cv2.imread(str(RAW_DIR / 'train' / fname))
    if img is None: continue
    axes[label_id].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[label_id].set_title(f'[{label_id}] {label_name}', fontsize=11, fontweight='bold')
    axes[label_id].axis('off'); shown[label_name] = True
    if all(shown.values()): break
plt.suptitle('One Raw Sample per Location Class (Ground Truth)', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()
"""))

# ── Cell 4 — Model architecture ─────────────────────────────────────────────
cells.append(code("""# Cell 4 — EfficientNet-B0 Architecture Summary
from claimvision_ml.location.efficientnet import build_location_efficientnet

model = build_location_efficientnet(dropout=0.3, num_classes=5, pretrained=True).to(device)
counts = model.parameter_count()
print('=== Location EfficientNet-B0 Architecture Summary ===')
print(f'Backbone      : EfficientNet-B0 via timm (ImageNet pretrained)')
print(f'Input Size    : 224 × 224 × 3 RGB')
print(f'Head          : Dropout → Linear(1280→128) → ReLU → Dropout → Linear(128→5)')
print(f'Classes (nc)  : 5 — {LOCATION_CLASSES}')
print(f'Device        : {device}')
print()
print(f'Total params  : {counts[\"total\"]:,}')
print(f'Trainable     : {counts[\"trainable\"]:,}  (Stage A: head only)')
print(f'Frozen        : {counts[\"frozen\"]:,}')
"""))

# ── Cell 5 — DataLoaders with Stratified Split ───────────────────────────────
cells.append(code("""# Cell 5 — Dataset and DataLoader (same stratified split as NB 13)
from torch.utils.data import DataLoader
from claimvision_ml.location.dataset import load_location_splits

BATCH_SIZE = 8; NUM_WORKERS = 0

train_ds, val_ds = load_location_splits(RAW_DIR, val_ratio=0.20, seed=SEED)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print('=== Split Construction Summary ===')
print(repr(train_ds))
print(repr(val_ds))
print(f'\\nTrain batches : {len(train_loader)} ({len(train_ds)} samples)')
print(f'Val batches   : {len(val_loader)} ({len(val_ds)} samples)')

print('\\nPer-class breakdown in validation set:')
val_counts = val_ds.label_counts
for cid, cname in enumerate(LOCATION_CLASSES):
    print(f'  [{cid}] {cname:<16}: {val_counts.get(cid, 0):>2} images')

class_weights = train_ds.class_weights().to(device)
print(f'\\nNormalized class weights: {[round(w, 3) for w in class_weights.tolist()]}')
"""))

# ── Cell 6 — Stage A ────────────────────────────────────────────────────────
cells.append(code("""# Cell 6 — Stage A: Train classification head (EfficientNet backbone frozen)
import torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

STAGE_A_EPOCHS = 25; STAGE_A_LR = 1e-3; PATIENCE = 12

criterion  = nn.CrossEntropyLoss(weight=class_weights)
optimizer  = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=STAGE_A_LR, weight_decay=1e-4)
scheduler  = CosineAnnealingLR(optimizer, T_max=STAGE_A_EPOCHS, eta_min=1e-5)

output_dir = REPO_ROOT / 'ml' / 'results' / 'location'
output_dir.mkdir(parents=True, exist_ok=True)
best_ckpt  = output_dir / 'location_efficientnet_best.pt'

history_a  = {'train_loss': [], 'val_loss': [], 'val_acc': []}
best_val_acc = -1.0
best_val_loss = float('inf')
patience_count = 0

print(f'Stage A — Training head for up to {STAGE_A_EPOCHS} epochs')
for epoch in range(1, STAGE_A_EPOCHS + 1):
    model.train(); train_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(); loss = criterion(model(imgs), labels); loss.backward(); optimizer.step()
        train_loss += loss.item() * imgs.size(0)
    train_loss /= len(train_loader.dataset)
    model.eval(); val_loss = correct = total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device); logits = model(imgs)
            val_loss += criterion(logits, labels).item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item(); total += imgs.size(0)
    val_loss /= len(val_loader.dataset); val_acc = correct / total
    history_a['train_loss'].append(train_loss); history_a['val_loss'].append(val_loss); history_a['val_acc'].append(val_acc)
    scheduler.step()

    if epoch == 1 or val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
        best_val_acc = val_acc; best_val_loss = val_loss; patience_count = 0
        torch.save(model.state_dict(), best_ckpt)
    else:
        patience_count += 1

    if epoch % 5 == 0 or epoch == 1:
        print(f'Epoch {epoch:>3}/{STAGE_A_EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {val_acc:.3f} (Best: {best_val_acc:.3f})')
    if patience_count >= PATIENCE: print(f'Early stop epoch {epoch}'); break
print(f'Stage A complete. Best val acc: {best_val_acc:.4f} (Val Loss: {best_val_loss:.4f})')
"""))

# ── Cell 7 — Stage A curves ─────────────────────────────────────────────────
cells.append(code("""# Cell 7 — Stage A Training Curves
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
e = range(1, len(history_a['train_loss'])+1)
axes[0].plot(e, history_a['train_loss'], label='Train Loss'); axes[0].plot(e, history_a['val_loss'], label='Val Loss')
axes[0].set_title('Stage A — Loss (EfficientNet-B0)'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(e, history_a['val_acc'], color='green'); axes[1].set_title('Stage A — Val Accuracy'); axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(output_dir / 'location_efficientnet_stage_a_curves.png', dpi=200); plt.show()
"""))

# ── Cell 8 — Stage B ────────────────────────────────────────────────────────
cells.append(code("""# Cell 8 — Stage B: Unfreeze last 2 EfficientNet blocks and fine-tune
STAGE_B_EPOCHS = 15; STAGE_B_LR = 5e-5; PATIENCE_B = 10

model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
model.unfreeze_last_blocks(n_blocks=2)
print(f'Stage B trainable params: {model.parameter_count()[\"trainable\"]:,}')
optimizer_b = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=STAGE_B_LR, weight_decay=1e-4)
scheduler_b = CosineAnnealingLR(optimizer_b, T_max=STAGE_B_EPOCHS, eta_min=1e-6)

history_b = {'train_loss': [], 'val_loss': [], 'val_acc': []}
best_b_acc = best_val_acc
best_b_loss = best_val_loss
patience_count_b = 0

for epoch in range(1, STAGE_B_EPOCHS + 1):
    model.train(); train_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer_b.zero_grad(); loss = criterion(model(imgs), labels); loss.backward(); optimizer_b.step()
        train_loss += loss.item() * imgs.size(0)
    train_loss /= len(train_loader.dataset)
    model.eval(); val_loss = correct = total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device); logits = model(imgs)
            val_loss += criterion(logits, labels).item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item(); total += imgs.size(0)
    val_loss /= len(val_loader.dataset); val_acc = correct / total
    history_b['train_loss'].append(train_loss); history_b['val_loss'].append(val_loss); history_b['val_acc'].append(val_acc)
    scheduler_b.step()
    if val_acc > best_b_acc or (val_acc == best_b_acc and val_loss < best_b_loss):
        best_b_acc = val_acc; best_b_loss = val_loss; patience_count_b = 0
        torch.save(model.state_dict(), best_ckpt)
    else:
        patience_count_b += 1
    if epoch % 5 == 0 or epoch == 1:
        print(f'Epoch {epoch:>3}/{STAGE_B_EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {val_acc:.3f} (Best: {best_b_acc:.3f})')
    if patience_count_b >= PATIENCE_B: print(f'Early stop epoch {epoch}'); break
print(f'Stage B complete. Best val acc: {best_b_acc:.4f} (Val Loss: {best_b_loss:.4f})')
"""))

# ── Cell 9 — Stage B curves ─────────────────────────────────────────────────
cells.append(code("""# Cell 9 — Stage B Training Curves
e = range(1, len(history_b['train_loss'])+1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(e, history_b['train_loss'], label='Train Loss'); axes[0].plot(e, history_b['val_loss'], label='Val Loss')
axes[0].set_title('Stage B — Loss (EfficientNet-B0 Last 2 Blocks)'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(e, history_b['val_acc'], color='green'); axes[1].set_title('Stage B — Val Accuracy'); axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(output_dir / 'location_efficientnet_stage_b_curves.png', dpi=200); plt.show()
"""))

# ── Cell 10 — Validation metrics ────────────────────────────────────────────
cells.append(code("""# Cell 10 — Validation: Per-class Metrics and Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True)); model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        preds = model(imgs.to(device)).argmax(1).cpu().tolist()
        all_preds.extend(preds); all_labels.extend(labels.tolist())

labels_list = list(range(len(LOCATION_CLASSES)))

print('=== Validation Classification Report ===')
print(classification_report(
    all_labels,
    all_preds,
    labels=labels_list,
    target_names=LOCATION_CLASSES,
    digits=4,
    zero_division=0,
))

cm = confusion_matrix(all_labels, all_preds, labels=labels_list)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=LOCATION_CLASSES, yticklabels=LOCATION_CLASSES, ax=ax)
ax.set_xlabel('Predicted'); ax.set_ylabel('Ground Truth')
ax.set_title('Validation Confusion Matrix — Location EfficientNet-B0')
plt.tight_layout(); plt.savefig(output_dir / 'location_efficientnet_val_cm.png', dpi=200); plt.show()
"""))

# ── Cell 11 — Visual inspection ─────────────────────────────────────────────
cells.append(code("""# Cell 11 — Visual Inspection: Correct and Incorrect Validation Predictions
from claimvision_ml.location.inference import classify_location

model.eval()
correct_examples, wrong_examples = [], []
for full_path, label_id in val_ds.samples[:20]:
    label_name = LOCATION_CLASSES[label_id]
    result = classify_location(full_path, model, model_type='efficientnet', device=device)
    entry = (full_path, label_name, result.class_name, result.confidence)
    (correct_examples if result.class_name == label_name else wrong_examples).append(entry)

def show_examples(examples, title, max_n=4):
    if not examples: print(f'No {title.lower()} examples.'); return
    n = min(max_n, len(examples))
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1: axes = [axes]
    for ax, (fpath, gt, pred, conf) in zip(axes, examples[:n]):
        img = cv2.imread(str(fpath))
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        color = 'green' if gt == pred else 'red'
        ax.set_title(f'GT: {gt}\\nPred: {pred}\\n({conf:.0%})', color=color, fontsize=9); ax.axis('off')
    plt.suptitle(title, fontsize=12, fontweight='bold'); plt.tight_layout(); plt.show()

show_examples(correct_examples, 'Correct Validation Predictions (green = correct class)')
show_examples(wrong_examples,   'Incorrect Validation Predictions (red = wrong class)')
"""))

# ── Cell 12 — Error analysis ────────────────────────────────────────────────
cells.append(code("""# Cell 12 — Error Analysis
print('=== Location EfficientNet-B0 — Qualitative Error Modes ===')
print()
print('Same error categories expected as MobileNetV2 (small dataset size):')
print('1. Front vs. rear bumper confusion — similar shape profile.')
print('2. Hood vs. front bumper on front-end collision images.')
print('3. Small headlamp area dominated by other parts in the frame.')
print()
print('EfficientNet-B0 advantage: compound scaling may better capture fine')
print('structural features (reflectors, grille patterns) that distinguish parts.')
print()
print('Results for comparison stored in: ml/results/location/')
"""))

# ── Cell 13 — Test evaluation ───────────────────────────────────────────────
cells.append(code("""# Cell 13 — Held-Out Test Set Forward Inference (unannotated test split)
print('=== HELD-OUT TEST SET FORWARD INFERENCE ===')
print('NOTE: The COCO test split contains NO ground-truth labels.')
print('Running purely in forward-inference mode to generate qualitative visual predictions.\\n')

test_img_dir = RAW_DIR / 'test'
test_images = sorted([f for f in test_img_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]) if test_img_dir.exists() else []
print(f'Found {len(test_images)} held-out unannotated test images.\\n')

if test_images:
    n_display = min(8, len(test_images))
    rows = (n_display + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for idx, img_path in enumerate(test_images[:n_display]):
        result = classify_location(img_path, model, model_type='efficientnet', device=device)
        top2_alt = ', '.join(f'{cls}({p:.0%})' for cls, p in result.top3[1:3])
        print(f'  [{idx+1}] {img_path.name:<15}: Pred = {result.class_name:<14} ({result.confidence:5.1%}) | Alt: {top2_alt}')
        img = cv2.imread(str(img_path))
        if img is not None:
            axes[idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[idx].set_title(f'{img_path.name}\\n{result.class_name} ({result.confidence:.0%})', fontsize=9)
            axes[idx].axis('off')

    for ax in axes[n_display:]:
        ax.axis('off')

    plt.suptitle('Held-Out Test Set — Qualitative Location Predictions (Unannotated)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    test_fig_path = output_dir / 'location_efficientnet_test_predictions.png'
    plt.savefig(test_fig_path, dpi=200)
    plt.show()
    print(f'\\nTest predictions plot saved: {test_fig_path}')
"""))

# ── Cell 14 — Latency benchmark ─────────────────────────────────────────────
cells.append(code("""# Cell 14 — Inference Latency & Model Size
sample_img = test_images[0] if test_images else val_ds.samples[0][0]
latency_ms = model.measure_cpu_latency(sample_img, num_runs=15)
ckpt_size_mb = best_ckpt.stat().st_size / (1024 * 1024) if best_ckpt.exists() else 20.0
print(f'Architecture   : EfficientNet-B0 (5 location classes)')
print(f'CPU Latency    : {latency_ms:.2f} ms/image')
print(f'Model Size     : {ckpt_size_mb:.2f} MB')
assert latency_ms < 250.0, f'Latency {latency_ms:.2f} ms exceeds acceptable limit'
print('PASS: Latency within acceptable range.')
"""))

# ── Cell 15 — Export ────────────────────────────────────────────────────────
cells.append(code("""# Cell 15 — Export artifacts (PyTorch + ONNX)
import shutil
from claimvision_ml.location.inference import export_location_onnx

models_dir = REPO_ROOT / 'artifacts' / 'models'
models_dir.mkdir(parents=True, exist_ok=True)

pt_dst = models_dir / 'location_efficientnet.pt'
shutil.copy2(str(best_ckpt), str(pt_dst)); print(f'PyTorch: {pt_dst}')

model.load_state_dict(torch.load(best_ckpt, map_location='cpu', weights_only=True)); model.eval().cpu()
onnx_path = export_location_onnx(model, models_dir / 'location_efficientnet.onnx')
print(f'ONNX   : {onnx_path} ({onnx_path.stat().st_size / 1024 / 1024:.2f} MB)')
"""))

# ── Cell 16 — Reproducibility ───────────────────────────────────────────────
cells.append(code("""# Cell 16 — Reproducibility Record
import hashlib, platform, datetime

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''): h.update(chunk)
    return h.hexdigest()

print('=== Reproducibility Record ===')
print(f'Python  : {platform.python_version()}  |  Seed: {SEED}')
print(f'Date    : {datetime.datetime.now().isoformat()}')
import torchvision, timm
print(f'torch   : {torch.__version__} | torchvision: {torchvision.__version__} | timm: {timm.__version__}')
if best_ckpt.exists(): print(f'SHA256  : {sha256_of(best_ckpt)}')
"""))

# ── Cell 17 — Go/No-Go ──────────────────────────────────────────────────────
cells.append(md("""## Go / No-Go Protocol — LOC-EFF-001

| Condition | Threshold | Decision |
|---|---|---|
| Val Accuracy ≥ 0.50 | ≥ 50 % | Proceed to comparison Notebook 15 |
| Val Accuracy 0.30–0.50 | 30–50 % | Experimental — compare with MobileNetV2 |
| Val Accuracy < 0.30 | < 30 % | No-Go — chance level; do not use in demo |

Compare directly with Notebook 13 (MobileNetV2) results in Notebook 15.
"""))

# ── Write ────────────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}
NB_PATH.parent.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Notebook 14 written to {NB_PATH} ({len(cells)} cells)")
