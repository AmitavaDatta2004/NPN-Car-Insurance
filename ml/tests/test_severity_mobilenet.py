"""Unit and integration tests for Severity MobileNetV2 and dataset utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from claimvision_ml.severity import (
    SEVERITY_CLASSES,
    SeverityDataset,
    SeverityMobileNetV2,
    SeverityResult,
    build_severity_mobilenet,
    export_onnx,
    get_severity_class_weights,
    get_severity_transforms,
    load_severity_checkpoint,
    predict_severity,
    save_severity_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_MANIFEST = REPO_ROOT / "data" / "manifests" / "severity_train.csv"
VAL_MANIFEST = REPO_ROOT / "data" / "manifests" / "severity_val.csv"
SAMPLE_IMAGE = REPO_ROOT / "data" / "raw" / "car_damage_severity" / "data3a" / "training" / "01-minor" / "0001.JPEG"


class TestSeverityMobileNetV2:
    def test_model_build_and_forward(self) -> None:
        model = build_severity_mobilenet(pretrained=False, num_classes=3)
        assert isinstance(model, SeverityMobileNetV2)
        assert model.num_classes == 3

        dummy_input = torch.randn(2, 3, 224, 224)
        logits = model(dummy_input)
        assert logits.shape == (2, 3)

    def test_freeze_and_unfreeze_stages(self) -> None:
        model = build_severity_mobilenet(pretrained=False, num_classes=3)

        # Stage A: Freeze backbone
        model.freeze_backbone()
        for p in model.features.parameters():
            assert not p.requires_grad
        for p in model.classifier.parameters():
            assert p.requires_grad

        counts_a = model.trainable_parameter_count()
        assert counts_a["trainable"] == sum(p.numel() for p in model.classifier.parameters())
        assert counts_a["frozen"] > 0

        # Stage B: Unfreeze last 2 blocks
        model.unfreeze_final_blocks(n_blocks=2)
        counts_b = model.trainable_parameter_count()
        assert counts_b["trainable"] > counts_a["trainable"]
        assert counts_b["frozen"] > 0

        # Check features[17] and features[18] have requires_grad=True
        assert any(p.requires_grad for p in model.features[17].parameters())
        assert any(p.requires_grad for p in model.features[18].parameters())
        # Earlier block still frozen
        assert not any(p.requires_grad for p in model.features[0].parameters())

    def test_checkpoint_roundtrip(self, tmp_path: Path) -> None:
        model = build_severity_mobilenet(pretrained=False, num_classes=3)
        ckpt_path = tmp_path / "test_severity_model.pt"

        saved_metrics = {"val_macro_f1": 0.85, "epoch": 5}
        save_severity_checkpoint(
            model=model,
            save_path=ckpt_path,
            epoch=5,
            metrics=saved_metrics,
        )
        assert ckpt_path.is_file()

        loaded_model, meta = load_severity_checkpoint(ckpt_path)
        assert isinstance(loaded_model, SeverityMobileNetV2)
        assert meta["epoch"] == 5
        assert meta["metrics"]["val_macro_f1"] == 0.85
        assert meta["classes"] == list(SEVERITY_CLASSES)

        # Compare outputs with both models in eval mode
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out1 = model(dummy_input)
            out2 = loaded_model(dummy_input)
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_onnx_export(self, tmp_path: Path) -> None:
        pytest.importorskip("onnx")
        model = build_severity_mobilenet(pretrained=False, num_classes=3)
        onnx_path = tmp_path / "test_severity.onnx"

        exported_path = export_onnx(model, onnx_path, verify=False)
        assert exported_path.is_file()
        assert exported_path.stat().st_size > 1_000_000  # >1MB


class TestSeverityDataset:
    def test_dataset_manifest_counts(self) -> None:
        if not VAL_MANIFEST.is_file():
            pytest.skip("Validation manifest not found")

        dataset = SeverityDataset(VAL_MANIFEST, transform=get_severity_transforms("val"))
        assert len(dataset) == 243

        counts = dataset.get_class_counts()
        assert counts["minor"] == 80
        assert counts["moderate"] == 80
        assert counts["severe"] == 83

    def test_dataset_getitem(self) -> None:
        if not VAL_MANIFEST.is_file():
            pytest.skip("Validation manifest not found")

        dataset = SeverityDataset(VAL_MANIFEST, transform=get_severity_transforms("val"))
        try:
            img_tensor, label_id, meta = dataset[0]
        except FileNotFoundError:
            pytest.skip("Severity raw image files not found on disk")

        assert isinstance(img_tensor, torch.Tensor)
        assert img_tensor.shape == (3, 224, 224)
        assert label_id in (0, 1, 2)
        assert meta["label"] in SEVERITY_CLASSES

    def test_transforms(self) -> None:
        img = Image.new("RGB", (320, 240), color=(100, 150, 200))
        t_train = get_severity_transforms("train", img_size=224)
        t_val = get_severity_transforms("val", img_size=224)

        train_out = t_train(img)
        val_out = t_val(img)

        assert train_out.shape == (3, 224, 224)
        assert val_out.shape == (3, 224, 224)

    def test_class_weights(self) -> None:
        if not TRAIN_MANIFEST.is_file():
            pytest.skip("Train manifest not found")

        weights = get_severity_class_weights(TRAIN_MANIFEST)
        assert isinstance(weights, torch.Tensor)
        assert weights.shape == (3,)
        assert (weights > 0).all()
        assert pytest.approx(float(weights.mean()), rel=1e-3) == 1.0


class TestSeverityPrediction:
    def test_predict_severity_synthetic_array(self) -> None:
        model = build_severity_mobilenet(pretrained=False, num_classes=3)
        img_arr = np.random.randint(0, 256, (250, 250, 3), dtype=np.uint8)

        res = predict_severity(img_arr, model)
        assert isinstance(res, SeverityResult)
        assert res.predicted_class in SEVERITY_CLASSES
        assert res.predicted_id in (0, 1, 2)
        assert 0.0 <= res.confidence <= 1.0
        assert len(res.probabilities) == 3
        prob_sum = sum(res.probabilities.values())
        assert pytest.approx(prob_sum, abs=1e-3) == 1.0

    def test_predict_severity_real_image(self) -> None:
        if not SAMPLE_IMAGE.is_file():
            pytest.skip("Sample test image not found")

        model = build_severity_mobilenet(pretrained=False, num_classes=3)
        res = predict_severity(SAMPLE_IMAGE, model)

        assert isinstance(res, SeverityResult)
        assert res.predicted_class in SEVERITY_CLASSES
        assert res.inference_time_ms > 0.0
