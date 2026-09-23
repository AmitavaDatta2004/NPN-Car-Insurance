"""Unit and integration tests for Severity MobileNetV2 and dataset utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from claimvision_ml.severity import (
    HybridOrdinalLoss,
    LetterboxResize,
    SEVERITY_CLASSES,
    SeverityDataset,
    SeverityMobileNetV2,
    SeverityResult,
    audit_batchnorm,
    build_letterbox_transforms,
    build_severity_mobilenet,
    compute_ordinal_error_metrics,
    export_onnx,
    get_severity_class_weights,
    get_severity_transforms,
    load_severity_checkpoint,
    make_class_weights,
    predict_severity,
    save_severity_checkpoint,
    set_frozen_batchnorm_eval,
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


class TestLetterboxPreprocessing:
    def test_letterbox_dimensions(self) -> None:
        # Wide aspect ratio
        wide_img = Image.new("RGB", (320, 160))
        lb_224 = LetterboxResize(224)
        out = lb_224(wide_img)
        assert out.size == (224, 224)

        # Tall aspect ratio
        tall_img = Image.new("RGB", (160, 320))
        lb_288 = LetterboxResize(288)
        out_tall = lb_288(tall_img)
        assert out_tall.size == (288, 288)

        # Square
        sq_img = Image.new("RGB", (200, 200))
        lb_320 = LetterboxResize(320)
        out_sq = lb_320(sq_img)
        assert out_sq.size == (320, 320)

    def test_build_letterbox_transforms(self) -> None:
        train_tf, eval_tf = build_letterbox_transforms(288)
        img = Image.new("RGB", (250, 180))

        eval_tensor = eval_tf(img)
        assert isinstance(eval_tensor, torch.Tensor)
        assert eval_tensor.shape == (3, 288, 288)
        assert eval_tensor.dtype == torch.float32

        train_tensor = train_tf(img)
        assert isinstance(train_tensor, torch.Tensor)
        assert train_tensor.shape == (3, 288, 288)

    def test_letterbox_invalid_dimensions(self) -> None:
        lb = LetterboxResize(224)
        with pytest.raises(ValueError, match="Invalid image dimensions"):
            lb(Image.new("RGB", (0, 100)))


class TestBatchNormAudit:
    def test_audit_batchnorm_structure(self) -> None:
        model = build_severity_mobilenet(pretrained=False, num_classes=3)
        model.freeze_backbone()
        model.unfreeze_final_blocks(4)
        model.train()

        audit = audit_batchnorm(model)
        assert "trainable_bn" in audit
        assert "frozen_bn_train" in audit
        assert "frozen_bn_eval" in audit
        assert isinstance(audit["trainable_bn"], list)
        assert isinstance(audit["frozen_bn_train"], list)
        assert isinstance(audit["frozen_bn_eval"], list)

        # In SeverityMobileNetV2, train() automatically keeps frozen BN in eval mode
        assert len(audit["trainable_bn"]) > 0
        assert len(audit["frozen_bn_eval"]) > 0

    def test_set_frozen_batchnorm_eval(self) -> None:
        model = build_severity_mobilenet(pretrained=False, num_classes=3)
        model.freeze_backbone()
        model.unfreeze_final_blocks(4)

        # Force all modules into train mode (bypassing overridden train)
        torch.nn.Module.train(model, True)
        audit_before = audit_batchnorm(model)
        assert len(audit_before["frozen_bn_train"]) > 0

        # Apply helper
        set_frozen_batchnorm_eval(model)
        audit_after = audit_batchnorm(model)
        assert len(audit_after["frozen_bn_train"]) == 0
        assert len(audit_after["frozen_bn_eval"]) == len(audit_before["frozen_bn_train"]) + len(audit_before["frozen_bn_eval"])


class TestPhase4And5LossAndMetrics:
    def test_make_class_weights(self) -> None:
        values = [0.9146, 1.1281, 0.9572]
        weights = make_class_weights(values)
        assert isinstance(weights, torch.Tensor)
        assert weights.shape == (3,)
        assert weights.dtype == torch.float32
        assert torch.isclose(weights.mean(), torch.tensor(1.0), atol=1e-5)

        uniform = make_class_weights([1.0, 1.0, 1.0])
        assert torch.allclose(uniform, torch.tensor([1.0, 1.0, 1.0]))

    def test_hybrid_ordinal_loss_forward_and_backward(self) -> None:
        weights = make_class_weights([1.0, 1.0, 1.0])
        criterion = HybridOrdinalLoss(class_weights=weights, label_smoothing=0.05, lambda_ordinal=0.2)

        logits = torch.randn(4, 3, requires_grad=True)
        labels = torch.tensor([0, 1, 2, 1], dtype=torch.long)

        total_loss, ce_loss, ordinal_loss = criterion(logits, labels)

        assert total_loss.item() > 0
        assert ce_loss.item() > 0
        assert ordinal_loss.item() >= 0
        assert torch.isclose(total_loss, ce_loss + 0.2 * ordinal_loss, atol=1e-5)

        total_loss.backward()
        assert logits.grad is not None
        assert logits.grad.shape == logits.shape

    def test_hybrid_ordinal_loss_zero_lambda(self) -> None:
        weights = make_class_weights([1.0, 1.0, 1.0])
        criterion = HybridOrdinalLoss(class_weights=weights, label_smoothing=0.05, lambda_ordinal=0.0)

        logits = torch.randn(4, 3)
        labels = torch.tensor([0, 1, 2, 1], dtype=torch.long)

        total_loss, ce_loss, _ = criterion(logits, labels)
        assert torch.isclose(total_loss, ce_loss)

    def test_compute_ordinal_error_metrics(self) -> None:
        # Perfect predictions
        preds_perfect = [0, 1, 2]
        labels_perfect = [0, 1, 2]
        m_perf = compute_ordinal_error_metrics(preds_perfect, labels_perfect)
        assert m_perf["severity_mae"] == 0.0
        assert m_perf["extreme_error_rate"] == 0.0
        assert m_perf["extreme_errors"] == 0
        assert m_perf["adjacent_errors"] == 0

        # Mixed predictions
        # 0 -> 1 (diff 1), 1 -> 2 (diff 1), 0 -> 2 (diff 2), 2 -> 0 (diff 2)
        preds = [0, 1, 0, 2]
        targets = [1, 2, 2, 0]
        m = compute_ordinal_error_metrics(preds, targets)
        # diffs: [1, 1, 2, 2] -> MAE = 1.5, extreme_rate = 0.5, extreme = 2, adjacent = 2
        assert m["severity_mae"] == 1.5
        assert m["extreme_error_rate"] == 0.5
        assert m["extreme_errors"] == 2
        assert m["adjacent_errors"] == 2
        assert m["minor_mod_errors"] == 1
        assert m["mod_sev_errors"] == 1

    def test_hybrid_ordinal_loss_device_and_dtype_alignment(self) -> None:
        weights = make_class_weights([1.0, 1.0, 1.0])
        criterion = HybridOrdinalLoss(class_weights=weights, label_smoothing=0.05, lambda_ordinal=0.2)

        # Test with float64 logits to ensure dtype/device conversion is handled seamlessly
        logits = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([0, 1, 2, 1], dtype=torch.long)

        total_loss, ce_loss, ordinal_loss = criterion(logits, labels)
        assert total_loss.dtype == torch.float64
        total_loss.backward()
        assert logits.grad is not None

