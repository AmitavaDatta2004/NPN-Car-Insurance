"""Audited, validation-only YOLO experiments; unlabelled test data is excluded."""

import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from .coco_converter import COCOtoYOLOConverter

PARTS = ["headlamp", "front_bumper", "hood", "door", "rear_bumper"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_coco(coco, image_dir, task):
    """Reject broken references, unknown classes and unreadable images before conversion."""
    names = ["damage"] if task == "damage" else PARTS
    if task not in {"damage", "parts"}:
        raise ValueError("Task must be damage or parts")
    result = COCOtoYOLOConverter.validate_structure(coco, image_dir)
    if not result.is_valid or result.warnings:
        raise ValueError(f"Invalid COCO annotations: {result}")
    if not coco["images"] or not coco["annotations"]:
        raise ValueError("A labelled split containing damage boxes is required")
    image_ids = [image["id"] for image in coco["images"]]
    stems = [Path(image["file_name"]).stem for image in coco["images"]]
    if len(set(image_ids)) != len(image_ids) or len(set(stems)) != len(stems):
        raise ValueError("Duplicate image IDs or output label filenames")
    mapping = {}
    for cat in coco["categories"]:
        name = cat["name"].strip().lower().replace(" ", "_")
        if name not in names:
            raise ValueError(f"Unexpected {task} category: {name}")
        if cat["id"] in mapping:
            raise ValueError("Duplicate category ID")
        mapping[cat["id"]] = names.index(name)
    for ann in coco["annotations"]:
        if ann["category_id"] not in mapping or ann["image_id"] not in image_ids:
            raise ValueError("Unknown annotation reference")
        if not np.isfinite(ann["bbox"]).all():
            raise ValueError("Nonfinite bounding box")
    fingerprints = {}
    for info in coco["images"]:
        path = Path(image_dir) / info["file_name"]
        if not path.is_file():
            path = Path(image_dir) / Path(info["file_name"]).name
        with Image.open(path) as image:
            image.load()
            if image.size != (info["width"], info["height"]):
                raise ValueError(
                    f"COCO dimensions differ from decoded image: {info['file_name']}"
                )
        fingerprints[info["file_name"]] = sha256(path)
    return mapping, fingerprints


def prepare_dataset(root, task, version="v1"):
    """Use Notebook 00 raw data; create fresh train/val conversion and provenance."""
    root = Path(root).resolve()
    if (
        task not in {"damage", "parts"}
        or Path(version).name != version
        or version in {".", ".."}
    ):
        raise ValueError("Invalid task or version")
    source_root = root / "data/raw/coco_car_damage"
    output = root / "artifacts/runs/improvements" / f"yolo_{task}_data_{version}"
    if output.exists():
        raise FileExistsError(
            f"Use a new dataset version; existing conversion is preserved: {output}"
        )
    records, splits = {}, {}
    for split in ("train", "val"):
        filename = f"COCO_{'mul_' if task == 'parts' else ''}{split}_annos.json"
        found = list(source_root.rglob(filename))
        if len(found) != 1:
            raise FileNotFoundError(
                f"Expected exactly one {filename} under Notebook 00's coco_car_damage folder"
            )
        source = found[0]
        coco = COCOtoYOLOConverter.load_coco_json(source)
        mapping, fingerprints = audit_coco(coco, source.parent, task)
        records[split] = {
            "annotation_sha256": sha256(source),
            "images": fingerprints,
            "image_count": len(coco["images"]),
            "box_count": len(coco["annotations"]),
        }
        splits[split] = (coco, source.parent, mapping)
    if set(records["train"]["images"].values()) & set(
        records["val"]["images"].values()
    ):
        raise ValueError(
            "Exact duplicate leakage between detection train and validation"
        )
    names = ["damage"] if task == "damage" else PARTS
    output.mkdir(parents=True)
    for split, (coco, image_dir, mapping) in splits.items():
        COCOtoYOLOConverter.convert_split(
            coco, image_dir, output / split, mapping, names
        )
        COCOtoYOLOConverter.run_conversion_assertions(output / split)
        records[split]["labels"] = {
            p.name: sha256(p) for p in (output / split / "labels").glob("*.txt")
        }
    # Intentionally no test key: the supplied eight test images lack ground truth.
    config = {
        "path": str(output),
        "train": "train/images",
        "val": "val/images",
        "nc": len(names),
        "names": names,
    }
    (output / "data.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "task": task,
                "splits": records,
                "test_metrics_available": False,
                "limitation": "Exact hash checks do not establish incident/source independence; inspect near duplicates and source groups.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output / "data.yaml"


def validate_training_yaml(path):
    """Only the audited dataset output is accepted by this experiment runner."""
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "test" in config:
        raise ValueError(
            "Improvement experiments accept train/val only; test evaluation is separate"
        )
    provenance = json.loads((path.parent / "provenance.json").read_text())
    expected_names = ["damage"] if provenance["task"] == "damage" else PARTS
    if config["names"] != expected_names or config["nc"] != len(expected_names):
        raise ValueError("Class mapping differs from audited conversion")
    base = Path(config["path"])
    if base.resolve() != path.parent.resolve():
        raise ValueError("Dataset path differs from audited conversion")
    fingerprints = []
    for split in ("train", "val"):
        if config[split] != f"{split}/images":
            raise ValueError("Dataset split differs from audited conversion")
        expected = provenance["splits"][split]["images"]
        image_dir = base / config[split]
        actual = {p.name: sha256(p) for p in image_dir.iterdir() if p.is_file()}
        if actual != {Path(name).name: value for name, value in expected.items()}:
            raise ValueError("Dataset images changed after audit")
        actual_labels = {
            p.name: sha256(p) for p in (base / split / "labels").glob("*.txt")
        }
        if actual_labels != provenance["splits"][split]["labels"]:
            raise ValueError("Dataset labels changed after audit")
        COCOtoYOLOConverter.run_conversion_assertions(base / split)
        fingerprints.append(set(actual.values()))
    if fingerprints[0] & fingerprints[1]:
        raise ValueError("Detection split leakage")
    return config, provenance


def run_experiment(root, data_yaml, run_id, variant="baseline", seed=42, epochs=100):
    """Compare existing YOLOv8n with a conservative-augmentation recipe."""
    from ultralytics import YOLO, __version__

    if variant not in {"baseline", "conservative"}:
        raise ValueError("Unknown YOLO recipe")
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("Invalid experiment ID")
    root = Path(root).resolve()
    _, provenance = validate_training_yaml(data_yaml)
    out = root / "artifacts/runs/improvements" / run_id
    out.mkdir(parents=True, exist_ok=False)
    settings = dict(
        data=str(Path(data_yaml).resolve()),
        epochs=epochs,
        patience=20,
        imgsz=640,
        batch=8,
        optimizer="AdamW",
        lr0=0.001,
        seed=seed,
        deterministic=True,
        workers=2,
        project=str(out),
        name="training",
        exist_ok=False,
    )
    if variant == "conservative":
        settings.update(
            mosaic=0.2,
            close_mosaic=10,
            mixup=0.0,
            degrees=5.0,
            translate=0.05,
            scale=0.2,
            fliplr=0.5,
            flipud=0.0,
            hsv_h=0.01,
            hsv_s=0.2,
            hsv_v=0.2,
        )
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    )
    record = {
        "run_id": run_id,
        "task_id": "ML-IMPROVE-001",
        "variant": variant,
        "git_commit": git.stdout.strip(),
        "ultralytics_version": __version__,
        "settings": settings,
        "dataset": provenance,
        "status": "RUNNING",
        "test_evaluated": False,
    }
    (out / "experiment.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    start = time.perf_counter()
    model = YOLO("yolov8n.pt")
    model.train(**settings)
    checkpoint = Path(model.trainer.save_dir) / "weights/best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "Training produced no best checkpoint; never substitute COCO weights"
        )
    selected = YOLO(str(checkpoint))
    metrics = selected.val(
        data=str(data_yaml),
        split="val",
        imgsz=640,
        plots=True,
        project=str(out),
        name="validation",
    )
    record.update(
        status="COMPLETE_VALIDATION_ONLY",
        checkpoint=str(checkpoint.relative_to(root)),
        checkpoint_sha256=sha256(checkpoint),
        duration_seconds=time.perf_counter() - start,
        validation={
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "per_class_ap": {
                str(int(i)): float(ap)
                for i, ap in zip(metrics.box.ap_class_index, metrics.box.ap)
            },
        },
        model_size_mb=checkpoint.stat().st_size / 2**20,
    )
    config = yaml.safe_load(Path(data_yaml).read_text())
    image = next((Path(config["path"]) / config["val"]).iterdir())
    # Real-image smoke inference, plus measured CPU timings (including preprocessing).
    timings = []
    for index in range(8):
        started = time.perf_counter()
        selected.predict(str(image), device="cpu", verbose=False)
        if index >= 3:
            timings.append((time.perf_counter() - started) * 1000)
    record["cpu_end_to_end_median_ms"] = float(np.median(timings))
    (out / "experiment.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
