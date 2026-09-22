"""Build Colab experiment notebooks; no training or fabricated outputs."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(source, kind="code"):
    result = {
        "cell_type": kind,
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def write(name, cells):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    (ROOT / "notebooks" / name).write_text(
        json.dumps(notebook, indent=1) + "\n", encoding="utf-8"
    )


BOOTSTRAP = """
from pathlib import Path
import sys, json, platform, random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from IPython.display import display

roots = [Path.cwd(), *Path.cwd().parents, Path('/content/NPN-Car-Insurance')]
ROOT = next((p for p in roots if (p/'ml/src/claimvision_ml').is_dir()), None)
if ROOT is None:
    raise RuntimeError('Run Notebook 00 first in this Colab runtime.')
sys.path.insert(0, str(ROOT/'ml/src'))
random.seed(42); np.random.seed(42); torch.manual_seed(42)
print({'python': platform.python_version(), 'torch': str(torch.__version__),
       'cuda': torch.cuda.is_available(), 'seed': 42})
if not torch.cuda.is_available():
    raise RuntimeError('Choose a Colab GPU runtime, then run Notebook 00 again.')
print('GPU:', torch.cuda.get_device_name(0))
"""


def main():
    write(
        "09b_severity_improvement_experiments.ipynb",
        [
            cell(
                """# Severity improvement experiments

Task: ML-IMPROVE-001 | Owner: Team lead / Codex | Date: 2026-09-22
Dataset: frozen Car Damage Severity v1 manifests, downloaded by Notebook 00.
Environment: Colab GPU with repository ML requirements. Status: NOT EXECUTED.

Question: does deeper fine-tuning improve validation macro F1 and accuracy, and does preserving the complete image add value?
Compare CNN, MobileNetV2, standard ViT-Tiny and the existing dual-stream ViT using the same train/validation membership.
The baseline here is a corrected reproducible control, not a reproduction of every historical hyperparameter.
`vit` and `vit_dual` have distinct checkpoint identities; the latter reuses Notebook 08's architecture with the controlled training protocol.
No test predictions are used. No default deployed model is overwritten.
Run Notebook 00 first. Do not regenerate the frozen manifests in Notebook 05.
""",
                "markdown",
            ),
            cell(BOOTSTRAP),
            cell("""
from claimvision_ml.severity.experiments import (
    Recipe, run_experiment, audit_manifests, compare_runs, load_candidate, PadSquare,
)
from claimvision_ml.severity.inputs import resolve_image, read_rgb
frames, checksums = audit_manifests(ROOT)
display(pd.DataFrame({s: f.label.value_counts() for s,f in frames.items()}))
print('Manifest SHA-256:', checksums)
print('Exact hashes and paths are disjoint. Existing audit supplies near-duplicate grouping.')
fig, axes = plt.subplots(3,2,figsize=(9,9))
for row, label in enumerate(['minor','moderate','severe']):
    item = frames['train'].query('label == @label').iloc[0]
    image = read_rgb(resolve_image(item.image_path, [ROOT]))
    axes[row,0].imshow(image); axes[row,0].set_title(label + ': original')
    axes[row,1].imshow(PadSquare()(image)); axes[row,1].set_title('Padding candidate (no crop)')
    for axis in axes[row]: axis.axis('off')
plt.tight_layout(); plt.show()
"""),
            cell(
                """## Controlled training

Three variants per architecture: `baseline` (partial backbone fine-tuning), `finetune` (all layers), `full_image` (same full fine-tuning plus padding instead of cropping). CNN has no pretrained backbone, so its baseline and finetune would be identical; run only baseline/full_image.
Pretrained heads warm up for five epochs, then use lower backbone learning rates. Frozen BatchNorm statistics stay frozen. Save the best validation macro F1 across all stages, restore it, and verify checkpoint reload predictions.
Allow up to 60 epochs with patience 12. CNN starts from scratch; MobileNet/ViT use ImageNet weights. Mean/std come from the ViT checkpoint configuration.
Run IDs are immutable: use a new prefix for a changed experiment. Start with seed 42; repeat promising configurations with seeds 43 and 44 before drawing conclusions.
""",
                "markdown",
            ),
            cell("""
RUN_PREFIX = 'severity-v2'
SEEDS = [42]  # Repeat finalists with [43, 44]; do not select a lucky seed.
run_ids = []
for architecture in ['cnn', 'mobilenet', 'vit', 'vit_dual']:
    variants = ['baseline','full_image'] if architecture == 'cnn' else ['baseline','finetune','full_image']
    for variant in variants:
        for seed in SEEDS:
            run_id = f'{RUN_PREFIX}-{architecture}-{variant}-s{seed}'
            record_path = ROOT/'artifacts/runs/improvements'/run_id/'experiment.json'
            recipe = Recipe(architecture, variant, seed=seed)
            if record_path.exists():
                record = json.loads(record_path.read_text())
                from dataclasses import asdict
                if record['status'] != 'COMPLETE_VALIDATION_ONLY' or record['recipe'] != asdict(recipe) or record['manifest_sha256'] != checksums:
                    raise RuntimeError('Existing incomplete or different run. Use a new RUN_PREFIX: ' + run_id)
                print('Reusing completed run:', run_id)
            else:
                record = run_experiment(ROOT, recipe, run_id)
            run_ids.append(run_id)
"""),
            cell("""
comparison = compare_runs(ROOT, run_ids)
display(comparison)
print('Group seed results; improvements must not come at the expense of unacceptable class recall.')
display(comparison.groupby(['architecture','variant'])[['val_macro_f1','val_accuracy','severe_recall']].agg(['mean','std']))
fig, axes = plt.subplots(1,2,figsize=(13,4))
for run_id in run_ids:
    h = pd.read_json(ROOT/'artifacts/runs/improvements'/run_id/'history.json')
    axes[0].plot(h.epoch,h.train_loss,label=run_id)
    axes[1].plot(h.epoch,h.macro_f1,label=run_id)
for ax, ylabel in zip(axes,['Training cross entropy','Validation macro F1']):
    ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel); ax.legend(fontsize=6)
plt.tight_layout(); plt.show()
"""),
            cell("""
# Validation examples only. This inspection is part of development, not final test evidence.
run_id = comparison.iloc[0].run_id
directory = ROOT/'artifacts/runs/improvements'/run_id
pred = np.load(directory/'validation_predictions.npz')
prob = torch.softmax(torch.tensor(pred['logits']),1).numpy()
guesses, labels = prob.argmax(1), pred['targets']
chosen = list(np.flatnonzero(guesses == labels)[:3]) + list(np.flatnonzero(guesses != labels)[:3])
fig, axes = plt.subplots(2,3,figsize=(12,7))
for ax in axes.flat: ax.axis('off')
classes = ['minor','moderate','severe']
for ax,index in zip(axes.flat,chosen):
    ax.imshow(read_rgb(resolve_image(str(pred['image_paths'][index]),[ROOT])))
    ax.set_title(f'True: {classes[labels[index]]} | Predicted: {classes[guesses[index]]} ({prob[index].max():.2f})')
plt.tight_layout(); plt.show()
record = json.loads((directory/'experiment.json').read_text())
fig, ax = plt.subplots(figsize=(5,4))
matrix = np.array(record['validation']['confusion_matrix'])
ax.imshow(matrix,cmap='Blues')
for i in range(3):
    for j in range(3): ax.text(j,i,str(matrix[i,j]),ha='center',va='center')
ax.set(xticks=range(3),yticks=range(3),xticklabels=classes,yticklabels=classes,
       xlabel='Predicted',ylabel='True',title='Validation confusion matrix')
plt.show()
print('Checkpoint:', directory/'best.pt')
print('Measured validation results:', record['validation'])
"""),
            cell("""
# Download local artifacts before Colab disconnects; includes private weights, not for Git.
import shutil
bundle = shutil.make_archive(str(ROOT/'severity_improvement_results'), 'zip',
                             ROOT/'artifacts/runs', 'improvements')
print(bundle)
if 'google.colab' in sys.modules:
    from google.colab import files
    files.download(bundle)
"""),
            cell(
                """## Findings, limitations and next decision

Fill in findings only after execution: which recipe improved each model, by how much, and which classes regressed? A candidate is not accepted simply because training completed. Keep the baseline if macro F1/accuracy or critical class recall regress.
Artifacts: `artifacts/runs/improvements/<run_id>/` contains weights, configuration, manifest checksums, history, validation predictions and metrics. Use `load_candidate` for exactly matching model/preprocessing; these candidates are not the legacy runtime checkpoints.
No final test claim is produced here. The historical test set has already been examined; independently collected labelled evidence is needed for a fresh generalisation claim. Report uncertainty from the small validation set, repeat seeds, and document the selected configuration before any final evaluation. No notebook completion or accuracy increase is claimed until restart/run-all succeeds in Colab.
""",
                "markdown",
            ),
        ],
    )
    write(
        "11b_detection_improvement_experiments.ipynb",
        [
            cell(
                """# Detection improvement experiments

Task: ML-IMPROVE-001 | Owner: Team lead / Codex | Date: 2026-09-22
Dataset: COCO Car Damage v1 from Notebook 00. Environment: Colab GPU. Status: NOT EXECUTED.
Question: does reducing destructive augmentation improve YOLOv8n localisation on this very small dataset?
Run generic damage first, then five damaged-part classes. New runs preserve existing models.
Only annotated train/validation splits are used. The eight unannotated test images cannot support mAP, precision or recall.
""",
                "markdown",
            ),
            cell(BOOTSTRAP),
            cell("""
from claimvision_ml.detection.experiments import prepare_dataset, validate_training_yaml, run_experiment
from PIL import Image
import matplotlib.patches as patches
import yaml
DATA_VERSION = 'v2'
data_paths = {}
for task in ['damage','parts']:
    path = ROOT/'artifacts/runs/improvements'/f'yolo_{task}_data_{DATA_VERSION}'/'data.yaml'
    if not path.exists(): path = prepare_dataset(ROOT, task, DATA_VERSION)
    config, provenance = validate_training_yaml(path)
    data_paths[task] = path
    print(task, {s: {'images': v['image_count'], 'boxes': v['box_count']} for s,v in provenance['splits'].items()})
    fig, axes = plt.subplots(1,3,figsize=(14,4))
    images = sorted((path.parent/'train/images').iterdir())[:3]
    for ax, image_path in zip(axes,images):
        image = Image.open(image_path); ax.imshow(image); ax.axis('off')
        for line in (path.parent/'train/labels'/f'{image_path.stem}.txt').read_text().splitlines():
            cls,cx,cy,w,h = map(float,line.split()); iw,ih = image.size
            x,y = (cx-w/2)*iw,(cy-h/2)*ih
            ax.add_patch(patches.Rectangle((x,y),w*iw,h*ih,fill=False,color='lime'))
            ax.text(x,y,config['names'][int(cls)],color='yellow')
    fig.suptitle(task + ': inspect converted ground-truth boxes before training')
    plt.show()
"""),
            cell(
                """## Train controlled comparisons

Inspect the grids above before running this cell. Stop if boxes or class names are wrong.
Baseline and candidate both use YOLOv8n, 640 pixels, AdamW, 100 maximum epochs and patience 20. Candidate: mosaic 0.2, no mixup, rotation 5 degrees, translation 0.05, scale 0.2, gentle colour jitter. These are hypotheses, not proven improvements.
Best checkpoints use the installed Ultralytics validation fitness rule; compare runs using validation mAP50–95 plus recall/per-class AP. Save the installed version and training args. Repeat finalists with additional seeds before promotion.
""",
                "markdown",
            ),
            cell("""
RUN_PREFIX = 'det-v2'
SEEDS = [42]
records = []
for task,path in data_paths.items():
    for variant in ['baseline','conservative']:
        for seed in SEEDS:
            run_id = f'{RUN_PREFIX}-{task}-{variant}-s{seed}'
            saved = ROOT/'artifacts/runs/improvements'/run_id/'experiment.json'
            if saved.exists():
                record = json.loads(saved.read_text())
                _, current = validate_training_yaml(path)
                if record['status'] != 'COMPLETE_VALIDATION_ONLY' or record['dataset'] != current:
                    raise RuntimeError('Use a new RUN_PREFIX for incomplete/changed runs: '+run_id)
            else:
                record = run_experiment(ROOT,path,run_id,variant,seed=seed)
            records.append(record)
display(pd.DataFrame([{'run':r['run_id'],**r['validation'],
                      'cpu_ms':r['cpu_end_to_end_median_ms']} for r in records]))
"""),
            cell("""
from ultralytics import YOLO
from IPython.display import Image as DisplayImage
for r in records:
    directory = ROOT/'artifacts/runs/improvements'/r['run_id']
    for plot in ['results.png','confusion_matrix.png']:
        candidates = list(directory.rglob(plot))
        if candidates: display(DisplayImage(filename=str(candidates[0]),width=700))
    task = r['dataset']['task']
    config = yaml.safe_load(data_paths[task].read_text())
    model = YOLO(str(ROOT/r['checkpoint']))
    paths = sorted((Path(config['path'])/'val/images').iterdir())[:3]
    fig,axes = plt.subplots(1,3,figsize=(14,4))
    for ax,p in zip(axes,paths):
        result = model.predict(str(p),verbose=False)[0]
        ax.imshow(result.plot()[...,::-1]); ax.axis('off'); ax.set_title(p.name)
    fig.suptitle(r['run_id'] + ': validation predictions (inspect misses/false alarms)')
    plt.show()
"""),
            cell("""
import shutil
bundle = shutil.make_archive(str(ROOT/'detection_improvement_results'),'zip',ROOT/'artifacts/runs','improvements')
print(bundle)
if 'google.colab' in sys.modules:
    from google.colab import files
    files.download(bundle)
"""),
            cell(
                """## Findings, limitations and next decision

After running, record whether validation mAP50–95, recall and per-class AP improved. Retain baseline weights if the candidate regresses. The part model remains experimental pending class-wise review.
59 training images and 11 validation images are insufficient to establish broad reliability. Inspect near duplicates and incident/source grouping: exact-hash isolation alone does not establish independence. Obtain more verified annotations before claiming generalisation. Unannotated test images are for qualitative inspection only; no test metric is produced.
Artifacts, provenance, library version, args and checksums are in `artifacts/runs/improvements/`. Download before the Colab runtime expires; weights and datasets stay out of Git. No measured improvement is claimed before execution.
""",
                "markdown",
            ),
        ],
    )


if __name__ == "__main__":
    main()
