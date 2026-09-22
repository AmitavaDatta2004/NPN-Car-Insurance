"""Regression tests use synthetic fixtures, never claimed model performance."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

from claimvision_ml.detection.experiments import (
    audit_coco,
    prepare_dataset,
    validate_training_yaml,
)
from claimvision_ml.severity import experiments as severity
from claimvision_ml.severity.inputs import resolve_image
from claimvision_ml.severity.vit import SeverityDataset


def test_best_checkpoint_preserves_early_stage_weights():
    model = nn.Linear(1, 1, bias=False)
    best = severity.BestCheckpoint()
    with torch.no_grad():
        model.weight.fill_(3)
    best.update(model, 0.6280, 3)
    with torch.no_grad():
        model.weight.fill_(5)
    assert not best.update(model, 0.5701, 5)
    assert not best.update(model, 0.6263, 14)
    best.restore(model)
    assert best.epoch == 3
    assert model.weight.item() == 3
    with pytest.raises(ValueError, match="Nonfinite"):
        best.update(model, float("nan"), 15)


def test_filename_only_fallback_cannot_cross_classes(tmp_path):
    for label in ("01-minor", "03-severe"):
        path = tmp_path / "data3a/training" / label / "0001.JPEG"
        path.parent.mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(path)
    raw = "data/raw/car_damage_severity/data3a/training/03-severe/0001.JPEG"
    assert resolve_image(raw, [tmp_path]).parent.name == "03-severe"
    with pytest.raises(FileNotFoundError):
        resolve_image("missing/0001.JPEG", [tmp_path])


@pytest.mark.parametrize("corrupt", [False, True])
def test_vit_never_substitutes_black_images(tmp_path, corrupt):
    path = tmp_path / "missing.jpg"
    if corrupt:
        path.write_bytes(b"not an image")
    dataset = SeverityDataset(pd.DataFrame([{"image_path": str(path), "label_id": 2}]))
    with pytest.raises((FileNotFoundError, ValueError)):
        dataset[0]


def test_frozen_batchnorm_statistics_do_not_change():
    model = nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(3, 3, 1))
    for p in model[0].parameters():
        p.requires_grad = False
    severity.train_mode(model)
    before = model[0].running_mean.clone()
    model(torch.randn(2, 3, 5, 5) + 10)
    torch.testing.assert_close(before, model[0].running_mean)
    assert model[1].training


def test_dual_stream_notebook_checkpoint_loads_in_runtime(tmp_path):
    from claimvision_ml.severity.dual_vit import DualStreamSeverityViT
    from claimvision_ml.severity.vit import load_vit_model, save_vit_checkpoint

    model = DualStreamSeverityViT(pretrained=False).eval()
    sample = torch.randn(1, 3, 224, 224)
    path = tmp_path / "dual.pt"
    save_vit_checkpoint(model, path, epoch=1)
    loaded = load_vit_model(path)
    with torch.inference_mode():
        torch.testing.assert_close(loaded(sample), model(sample), rtol=0, atol=0)


def test_metrics_expose_class_collapse():
    result = severity.metrics_from_logits(np.array([[0.0, 0.0, 20.0]] * 3), [0, 1, 2])
    assert result["accuracy"] == pytest.approx(1 / 3)
    assert result["macro_f1"] == pytest.approx(1 / 6)
    assert result["class_recall"] == [0, 0, 1]
    assert result["ece_10_bins"] > 0.6


def make_severity_data(root):
    manifest_dir = root / "data/manifests"
    manifest_dir.mkdir(parents=True)
    for index, split in enumerate(("train", "val", "test")):
        rows = []
        for label_id, label in enumerate(severity.CLASSES):
            rel = f"data/raw/car_damage_severity/{split}/{label}.png"
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (20, 16), (index * 50, label_id * 50, 10)).save(path)
            rows.append(
                {
                    "image_path": rel,
                    "label_id": label_id,
                    "label": label,
                    "sha256": severity.digest(path),
                }
            )
        pd.DataFrame(rows).to_csv(manifest_dir / f"severity_{split}.csv", index=False)


def test_frozen_manifest_leakage_and_checksum_guard(tmp_path):
    make_severity_data(tmp_path)
    frames, _ = severity.audit_manifests(tmp_path)
    Image.new("RGB", (20, 16), "white").save(
        tmp_path / frames["train"].iloc[0].image_path
    )
    with pytest.raises(ValueError, match="checksum"):
        severity.ManifestImages(frames["train"], tmp_path, severity.make_transform(16))
    val = tmp_path / "data/manifests/severity_val.csv"
    frame = pd.read_csv(val)
    frame.loc[0, "sha256"] = frames["train"].iloc[0].sha256
    frame.to_csv(val, index=False)
    with pytest.raises(ValueError, match="leakage"):
        severity.audit_manifests(tmp_path)


def test_training_smoke_never_reads_test_images_and_reloads_checkpoint(
    tmp_path, monkeypatch
):
    make_severity_data(tmp_path)
    # Remove test pixels: validation-only training must still succeed.
    for path in (tmp_path / "data/raw/car_damage_severity/test").glob("*.png"):
        path.unlink()
    monkeypatch.setattr(
        severity,
        "build_model",
        lambda *a, **kw: nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 3)
        ),
    )
    recipe = severity.Recipe("cnn", epochs=2, warmup_epochs=0, workers=0, batch_size=3)
    result = severity.run_experiment(tmp_path, recipe, "unit-smoke", pretrained=False)
    assert result["status"] == "COMPLETE_VALIDATION_ONLY"
    assert not result["test_evaluated"]
    model, transform = severity.load_candidate(
        tmp_path / "artifacts/runs/improvements/unit-smoke/best.pt"
    )
    assert model(transform(Image.new("RGB", (20, 20))).unsqueeze(0)).shape == (1, 3)
    assert len(severity.compare_runs(tmp_path, ["unit-smoke"])) == 1
    with pytest.raises(FileExistsError):
        severity.run_experiment(tmp_path, recipe, "unit-smoke", pretrained=False)


def coco_fixture(root, split, color):
    directory = root / "data/raw/coco_car_damage" / split
    directory.mkdir(parents=True)
    Image.new("RGB", (20, 16), color).save(directory / "car.png")
    coco = {
        "images": [{"id": 1, "file_name": "car.png", "width": 20, "height": 16}],
        "categories": [{"id": 1, "name": "damage"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [2, 2, 8, 8]}
        ],
    }
    (directory / f"COCO_{split}_annos.json").write_text(json.dumps(coco))
    return coco, directory


def test_detection_conversion_excludes_unlabelled_test_and_checks_tampering(tmp_path):
    coco_fixture(tmp_path, "train", "red")
    coco_fixture(tmp_path, "val", "blue")
    path = prepare_dataset(tmp_path, "damage")
    config, _ = validate_training_yaml(path)
    assert "test" not in config
    label = path.parent / "val/labels/car.txt"
    label.write_text("")
    with pytest.raises(ValueError, match="labels changed"):
        validate_training_yaml(path)


def test_detection_rejects_empty_annotations_and_unknown_categories(tmp_path):
    coco, directory = coco_fixture(tmp_path, "train", "red")
    coco["annotations"] = []
    with pytest.raises(ValueError, match="labelled split"):
        audit_coco(coco, directory, "damage")
    coco["annotations"] = [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [2, 2, 8, 8]}
    ]
    coco["categories"][0]["name"] = "unknown_part"
    with pytest.raises(ValueError, match="Unexpected"):
        audit_coco(coco, directory, "damage")


def test_detection_split_duplicates_rejected(tmp_path):
    coco_fixture(tmp_path, "train", "red")
    coco_fixture(tmp_path, "val", "red")
    with pytest.raises(ValueError, match="leakage"):
        prepare_dataset(tmp_path, "damage")


def test_mobilenet_notebook_stage_a_retains_epoch_three(tmp_path):
    # Exercise the real notebook's corrected Stage A cell with deterministic scores.
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads(
        (root / "notebooks/07_severity_mobilenetv2_training.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = next(
        "".join(c["source"])
        for c in notebook["cells"]
        if "Starting Stage A Training" in "".join(c["source"])
    )
    import copy
    import time

    model = nn.Module()
    model.classifier = nn.Linear(1, 1)
    scores = iter([0.5991, 0.5929, 0.6280, 0.5914, 0.5701, 0.5900])
    step = 0

    def train(*args):
        nonlocal step
        step += 1
        with torch.no_grad():
            model.classifier.weight.fill_(step)
        return 1.0, 0.5

    context = dict(
        torch=torch,
        copy=copy,
        time=time,
        model=model,
        train_loader=None,
        val_loader=None,
        criterion=None,
        DEVICE="cpu",
        REPO_ROOT=tmp_path,
        save_severity_checkpoint=lambda **kwargs: None,
        train_one_epoch=train,
        evaluate_model=lambda *a: dict(
            loss=1.0,
            accuracy=0.5,
            macro_f1=next(scores),
            severe_recall=0.5,
            moderate_recall=0.5,
            minor_recall=0.5,
        ),
    )
    exec(compile(source, "notebook-stage-a", "exec"), context)
    assert context["best_epoch"] == 3
    assert context["best_model_state"]["classifier.weight"].item() == 3
