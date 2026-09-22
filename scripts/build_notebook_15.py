#!/usr/bin/env python3
"""build_notebook_15.py — Generate notebooks/15_location_model_comparison.ipynb.

Task ID : LOC-COMP-001
Run     : python scripts/build_notebook_15.py
Output  : notebooks/15_location_model_comparison.ipynb
"""

from __future__ import annotations
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_PATH = REPO_ROOT / "notebooks" / "15_location_model_comparison.ipynb"

cells = []

def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source, "outputs": [], "execution_count": None}

def code(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": source, "outputs": [], "execution_count": None}

# ── Cell 0 ──────────────────────────────────────────────────────────────────
cells.append(md("""# Notebook 15 — Location Classification: Model Comparison

**Task ID:** LOC-COMP-001  
**Phase:** 11c — Location CNN Model Comparison  
**Owner:** Detection ML Member / Antigravity  
**Date:** 2026-09-22  
**Environment:** Google Colab (CPU is sufficient — no training here)

## Purpose

Compare the two location classifiers trained in Notebooks 13 and 14 on the **same held-out test images**  
and select the model to integrate into the unified inference demo (Notebook 16).

**Models compared:**

| Model | Notebook | Task ID | Architecture |
|---|---|---|---|
| MobileNetV2 | 13 | LOC-MNV2-001 | ImageNet MNV2 + custom head |
| EfficientNet-B0 | 14 | LOC-EFF-001 | timm EfficientNet-B0 + custom head |

**Selection criteria:** macro F1, per-class recall, CPU latency, model size, and qualitative failure analysis.

> **Scientific note:** The test set is unannotated (8 images, no ground-truth labels).  
> Comparison uses validation set metrics as the ground-truth proxy.  
> No model tuning is done here — models are loaded from frozen checkpoints.
"""))

# ── Cell 1 — Environment ────────────────────────────────────────────────────
cells.append(code("""# Cell 1 — Environment Setup
import os, sys, json
from pathlib import Path
import numpy as np
import torch

SEED = 42

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

device = 'cpu'  # Comparison only — GPU not needed
print(f'Environment: {\"Google Colab\" if IN_COLAB else \"Local\"} | Device: {device}')
print('NOTE: This notebook performs NO training. It loads checkpoints and evaluates.')
"""))

# ── Cell 2 — Load checkpoints ────────────────────────────────────────────────
cells.append(code("""# Cell 2 — Load both model checkpoints
from claimvision_ml.location.mobilenet import load_location_mobilenet
from claimvision_ml.location.efficientnet import load_location_efficientnet
from claimvision_ml.location.dataset import LOCATION_CLASSES

RESULTS_DIR = REPO_ROOT / 'ml' / 'results' / 'location'
MNV2_CKPT   = RESULTS_DIR / 'location_mobilenetv2_best.pt'
EFF_CKPT    = RESULTS_DIR / 'location_efficientnet_best.pt'

print('=== Loading checkpoints ===')
models = {}
if MNV2_CKPT.exists():
    models['MobileNetV2']     = load_location_mobilenet(MNV2_CKPT, device=device)
    print(f'  MobileNetV2      loaded: {MNV2_CKPT}')
else:
    print(f'  WARNING: MobileNetV2 checkpoint not found at {MNV2_CKPT}')
    print('  Run Notebook 13 first.')

if EFF_CKPT.exists():
    models['EfficientNet-B0'] = load_location_efficientnet(EFF_CKPT, device=device)
    print(f'  EfficientNet-B0  loaded: {EFF_CKPT}')
else:
    print(f'  WARNING: EfficientNet-B0 checkpoint not found at {EFF_CKPT}')
    print('  Run Notebook 14 first.')

print(f'\\n{len(models)} model(s) loaded for comparison.')
print(f'Classes: {LOCATION_CLASSES}')
"""))

# ── Cell 3 — Load validation dataset ────────────────────────────────────────
cells.append(code("""# Cell 3 — Load validation dataset (same split used in NB 13 & 14)
from claimvision_ml.location.dataset import LocationDataset, derive_location_labels
from torch.utils.data import DataLoader

RAW_CANDIDATES = [
    REPO_ROOT / 'data' / 'raw' / 'coco_car_damage' / 'coco-car-damage-detection-dataset' / 'coco-car-damage-detection-dataset',
    REPO_ROOT / 'data' / 'raw' / 'coco_car_damage' / 'coco-car-damage-detection-dataset',
    Path('/content/NPN-Car-Insurance/data/raw/coco_car_damage/coco-car-damage-detection-dataset/coco-car-damage-detection-dataset'),
]
RAW_DIR = next((c for c in RAW_CANDIDATES if (c / 'val').exists()), None)
if RAW_DIR is None:
    raise RuntimeError('COCO dataset not found. Run Notebook 00 first.')

val_ds = LocationDataset(RAW_DIR / 'val', RAW_DIR / 'val' / 'COCO_mul_val_annos.json', split='val')
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
print(repr(val_ds))
"""))

# ── Cell 4 — Evaluate both on validation ────────────────────────────────────
cells.append(code("""# Cell 4 — Evaluate both models on validation set
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import torch.nn.functional as F

def evaluate_model(model_obj, loader, model_name):
    model_obj.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model_obj(imgs.to('cpu'))
            all_preds.extend(logits.argmax(1).tolist())
            all_labels.extend(labels.tolist())
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    report = classification_report(all_labels, all_preds, target_names=LOCATION_CLASSES, digits=4, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)
    print(f'\\n=== {model_name} ===')
    print(f'Accuracy   : {acc:.4f}')
    print(f'Macro F1   : {macro_f1:.4f}')
    print(classification_report(all_labels, all_preds, target_names=LOCATION_CLASSES, digits=4))
    return {'name': model_name, 'macro_f1': macro_f1, 'accuracy': acc, 'report': report, 'cm': cm, 'preds': all_preds, 'labels': all_labels}

results = {}
for name, model_obj in models.items():
    results[name] = evaluate_model(model_obj, val_loader, name)
"""))

# ── Cell 5 — Comparison table ────────────────────────────────────────────────
cells.append(code("""# Cell 5 — Side-by-Side Metric Comparison Table
import pandas as pd

rows = []
for name, res in results.items():
    rpt = res['report']
    row = {
        'Model': name,
        'Accuracy': f\"{res['accuracy']:.4f}\",
        'Macro F1': f\"{res['macro_f1']:.4f}\",
    }
    for cls in LOCATION_CLASSES:
        row[f'{cls[:8]} F1'] = f\"{rpt.get(cls, {}).get('f1-score', 0):.4f}\"
    rows.append(row)

df = pd.DataFrame(rows).set_index('Model')
print('=== Location Model Comparison (Validation Set) ===')
print(df.to_string())
print()
print('NOTE: Test set is unannotated (8 images) — validation metrics are the primary comparison basis.')
print('NOTE: Both models trained on only ~59 images — results are prototype-quality, not production.')
"""))

# ── Cell 6 — Side-by-side confusion matrices ────────────────────────────────
cells.append(code("""# Cell 6 — Side-by-Side Confusion Matrices
import matplotlib.pyplot as plt
import seaborn as sns

n_models = len(results)
fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 6))
if n_models == 1:
    axes = [axes]

for ax, (name, res) in zip(axes, results.items()):
    cm = res['cm']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LOCATION_CLASSES, yticklabels=LOCATION_CLASSES, ax=ax)
    ax.set_title(f'{name}\\nMacro F1={res[\"macro_f1\"]:.4f}', fontsize=12)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Ground Truth')
    for tick in ax.get_xticklabels():
        tick.set_rotation(30); tick.set_ha('right')

plt.suptitle('Location Classifier Confusion Matrices — Validation Set', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(REPO_ROOT / 'ml' / 'results' / 'location' / 'location_comparison_cm.png', dpi=200)
plt.show()
"""))

# ── Cell 7 — Per-class F1 bar chart ─────────────────────────────────────────
cells.append(code("""# Cell 7 — Per-Class F1 Bar Chart
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(len(LOCATION_CLASSES))
width = 0.35
colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']

fig, ax = plt.subplots(figsize=(10, 5))
for i, (name, res) in enumerate(results.items()):
    f1s = [res['report'].get(cls, {}).get('f1-score', 0) for cls in LOCATION_CLASSES]
    ax.bar(x + i * width - (len(results)-1)*width/2, f1s, width, label=name, alpha=0.8, color=colors[i])

ax.set_xticks(x); ax.set_xticklabels(LOCATION_CLASSES, rotation=20, ha='right')
ax.set_ylabel('F1 Score'); ax.set_ylim(0, 1)
ax.set_title('Per-Class F1 Score Comparison — Location Classifiers (Validation Set)')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(REPO_ROOT / 'ml' / 'results' / 'location' / 'location_comparison_f1_bar.png', dpi=200)
plt.show()
"""))

# ── Cell 8 — Latency and size ────────────────────────────────────────────────
cells.append(code("""# Cell 8 — CPU Latency and Model Size Comparison
from PIL import Image

sample = Image.new('RGB', (224, 224))

print('=== Latency & Model Size ===')
print(f'{\"Model\":<18} {\"Latency (ms)\":>14} {\"Size (MB)\":>12}')
print('-' * 46)
for name, model_obj in models.items():
    try:
        latency = model_obj.measure_cpu_latency(sample, num_runs=15)
    except AttributeError:
        latency = float('nan')

    ckpt_name = 'location_mobilenetv2_best.pt' if 'Mobile' in name else 'location_efficientnet_best.pt'
    ckpt_path = REPO_ROOT / 'ml' / 'results' / 'location' / ckpt_name
    size_mb = ckpt_path.stat().st_size / (1024 * 1024) if ckpt_path.exists() else float('nan')

    print(f'{name:<18} {latency:>14.2f} {size_mb:>12.2f}')
"""))

# ── Cell 9 — Shared difficult examples ──────────────────────────────────────
cells.append(code("""# Cell 9 — Shared Difficult Examples (images both models got wrong)
import cv2

label_map = derive_location_labels(RAW_DIR / 'val' / 'COCO_mul_val_annos.json')
val_img_dir = RAW_DIR / 'val'

from claimvision_ml.location.inference import classify_location

shared_errors = []
for fname, (label_id, label_name) in label_map.items():
    img_path = val_img_dir / fname
    if not img_path.exists(): continue
    preds = {}
    for name, model_obj in models.items():
        result = classify_location(img_path, model_obj, model_type='mobilenet' if 'Mobile' in name else 'efficientnet')
        preds[name] = (result.class_name, result.confidence)
    if all(p != label_name for p, _ in preds.values()):
        shared_errors.append((img_path, label_name, preds))

print(f'Images both models got wrong: {len(shared_errors)}')
if shared_errors:
    n = min(4, len(shared_errors))
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1: axes = [axes]
    for ax, (fpath, gt, preds) in zip(axes, shared_errors[:n]):
        img = cv2.imread(str(fpath))
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pred_str = '\\n'.join(f'{k[:4]}: {v[0]}({v[1]:.0%})' for k, v in preds.items())
        ax.set_title(f'GT: {gt}\\n{pred_str}', color='red', fontsize=8); ax.axis('off')
    plt.suptitle('Shared Difficult Examples (Both Models Wrong)', fontsize=12); plt.tight_layout(); plt.show()
"""))

# ── Cell 10 — Test set inference ────────────────────────────────────────────
cells.append(code("""# Cell 10 — Qualitative Test Set Comparison (unannotated)
test_img_dir = RAW_DIR / 'test'
test_images = sorted(test_img_dir.glob('*.jpg'))
print(f'Comparing predictions on {len(test_images)} held-out test images.')
print()
print(f'{\"Image\":<25} {\"MobileNetV2\":>20} {\"EfficientNet-B0\":>22}')
print('-' * 70)
for img_path in test_images:
    row = {name: classify_location(img_path, m, model_type='mobilenet' if 'Mobile' in name else 'efficientnet')
           for name, m in models.items()}
    mnv2 = list(row.values())[0] if 'MobileNetV2' in row else None
    eff  = list(row.values())[-1] if 'EfficientNet-B0' in row else None
    mnv2_str = f'{mnv2.class_name} ({mnv2.confidence:.0%})' if mnv2 else 'N/A'
    eff_str  = f'{eff.class_name} ({eff.confidence:.0%})'  if eff  else 'N/A'
    agree = '✓' if (mnv2 and eff and mnv2.class_name == eff.class_name) else '✗'
    print(f'{img_path.name:<25} {mnv2_str:>20} {eff_str:>22}  {agree}')
"""))

# ── Cell 11 — Selection decision ────────────────────────────────────────────
cells.append(md("""## Model Selection Decision — LOC-COMP-001

Evaluate the printed metrics above and complete this table before committing:

| Criterion | MobileNetV2 | EfficientNet-B0 | Winner |
|---|---|---|---|
| Validation Accuracy | _fill in_ | _fill in_ | _fill in_ |
| Macro F1 | _fill in_ | _fill in_ | _fill in_ |
| CPU latency (ms) | _fill in_ | _fill in_ | _fill in_ |
| Model size (MB) | _fill in_ | _fill in_ | _fill in_ |
| **Overall** | | | **_fill in_** |

### Decision

> **Selected model:** _fill in after running notebook_  
> **Reason:** _fill in_  
> **Experiment ID:** LOC-COMP-001  
> **Date:** 2026-09-22

Record this decision in `docs/EXPERIMENT_LOG.md` and update `docs/MODEL_CARD_LOCATION_MNV2_V1.md` or  
`docs/MODEL_CARD_LOCATION_EFF_V1.md` with the final test metrics.

> [!IMPORTANT]
> **Per AGENTS.md §3:** The selected model must be justified by accuracy, class-wise behavior,  
> calibration, latency, size, and demo stability — **not accuracy alone**.
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
print(f"Notebook 15 written to {NB_PATH} ({len(cells)} cells)")
