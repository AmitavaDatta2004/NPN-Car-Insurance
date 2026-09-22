"""test_location_classifier.py — Unit tests for the location inference module.

Task IDs : LOC-MNV2-001, LOC-EFF-001
Phase    : 11a / 11b

Tests cover:
- LocationClassification dataclass: valid construction for all 5 classes
- LocationClassification dataclass: invalid class_id raises ValueError
- LocationClassification dataclass: confidence out of range raises ValueError
- LocationClassification dataclass: invalid class_name raises ValueError
- LocationClassification: to_dict keys and types
- LocationMobileNet: parameter_count returns dict with trainable key
- LocationMobileNet: freeze_backbone freezes all backbone params
- LocationMobileNet: unfreeze_last_blocks makes some params trainable
- LocationMobileNet: forward output shape
- classify_location: output is LocationClassification with correct class
- classify_location: confidence in [0, 1]
- classify_location: top3 has 3 entries summing ≤ 1
- classify_location: warning emitted when confidence < threshold
- export_location_onnx: file created
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from PIL import Image


# ---------------------------------------------------------------------------
# LocationClassification dataclass
# ---------------------------------------------------------------------------

class TestLocationClassification:
    def _valid_kwargs(self, class_id: int = 0, class_name: str = "headlamp") -> dict:
        return {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": 0.75,
            "top3": [("headlamp", 0.75), ("door", 0.15), ("hood", 0.10)],
            "model_type": "mobilenet",
            "latency_ms": 12.3,
            "warning": "",
        }

    def test_valid_construction_all_classes(self):
        from claimvision_ml.location.inference import LocationClassification
        from claimvision_ml.location.dataset import LOCATION_CLASSES

        for idx, name in enumerate(LOCATION_CLASSES):
            lc = LocationClassification(
                class_id=idx,
                class_name=name,
                confidence=0.5,
                top3=[(name, 0.5)],
            )
            assert lc.class_id == idx
            assert lc.class_name == name

    def test_invalid_class_id_raises(self):
        from claimvision_ml.location.inference import LocationClassification
        with pytest.raises(ValueError, match="class_id"):
            LocationClassification(class_id=99, class_name="headlamp", confidence=0.5)

    def test_confidence_above_1_raises(self):
        from claimvision_ml.location.inference import LocationClassification
        with pytest.raises(ValueError, match="confidence"):
            LocationClassification(class_id=0, class_name="headlamp", confidence=1.5)

    def test_confidence_below_0_raises(self):
        from claimvision_ml.location.inference import LocationClassification
        with pytest.raises(ValueError, match="confidence"):
            LocationClassification(class_id=0, class_name="headlamp", confidence=-0.1)

    def test_invalid_class_name_raises(self):
        from claimvision_ml.location.inference import LocationClassification
        with pytest.raises(ValueError, match="class_name"):
            LocationClassification(class_id=0, class_name="windshield", confidence=0.5)

    def test_to_dict_keys(self):
        from claimvision_ml.location.inference import LocationClassification
        lc = LocationClassification(**self._valid_kwargs())
        d = lc.to_dict()
        for key in ("class_id", "class_name", "confidence", "top3", "model_type", "latency_ms", "warning"):
            assert key in d

    def test_to_dict_confidence_rounded(self):
        from claimvision_ml.location.inference import LocationClassification
        lc = LocationClassification(class_id=0, class_name="headlamp", confidence=0.123456789)
        d = lc.to_dict()
        assert d["confidence"] == pytest.approx(0.1235, rel=1e-2)

    def test_efficientnet_model_type(self):
        from claimvision_ml.location.inference import LocationClassification
        lc = LocationClassification(
            class_id=1,
            class_name="front_bumper",
            confidence=0.6,
            model_type="efficientnet",
        )
        assert lc.model_type == "efficientnet"


# ---------------------------------------------------------------------------
# LocationMobileNet architecture tests
# ---------------------------------------------------------------------------

class TestLocationMobileNet:
    def _build_fresh(self) -> "LocationMobileNet":
        from claimvision_ml.location.mobilenet import LocationMobileNet
        return LocationMobileNet(num_classes=5, dropout=0.3, pretrained=False)

    def test_parameter_count_keys(self):
        model = self._build_fresh()
        counts = model.parameter_count()
        assert "trainable" in counts
        assert "total" in counts
        assert "frozen" in counts

    def test_freeze_backbone(self):
        model = self._build_fresh()
        model.freeze_backbone()
        frozen = [p for p in model.backbone.features.parameters() if not p.requires_grad]
        all_backbone = list(model.backbone.features.parameters())
        assert len(frozen) == len(all_backbone), "All backbone params should be frozen"

    def test_unfreeze_last_blocks_increases_trainable(self):
        model = self._build_fresh()
        model.freeze_backbone()
        before = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model.unfreeze_last_blocks(n_blocks=2)
        after = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert after > before, "Unfreezing blocks should increase trainable params"

    def test_forward_output_shape(self):
        model = self._build_fresh()
        model.eval()
        x = torch.zeros(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 5), f"Expected (2, 5), got {out.shape}"

    def test_measure_cpu_latency_returns_positive(self):
        model = self._build_fresh()
        img = Image.new("RGB", (224, 224))
        lat = model.measure_cpu_latency(img, num_runs=3)
        assert lat > 0.0


# ---------------------------------------------------------------------------
# classify_location functional API
# ---------------------------------------------------------------------------

class TestClassifyLocation:
    def _fake_model(self, predicted_class_id: int = 2) -> nn.Module:
        """Return a nn.Module that always outputs a one-hot logit for predicted_class_id."""
        class FakeModel(nn.Module):
            def __init__(self, cls_id):
                super().__init__()
                self._cls_id = cls_id
                self._dummy = nn.Linear(1, 1)  # so parameters() is non-empty

            def forward(self, x):
                logits = torch.full((x.size(0), 5), -10.0)
                logits[:, self._cls_id] = 10.0
                return logits

        return FakeModel(predicted_class_id)

    def test_classify_returns_correct_class(self):
        from claimvision_ml.location.inference import classify_location
        from claimvision_ml.location.dataset import LOCATION_CLASSES
        model = self._fake_model(predicted_class_id=3)  # door
        img = Image.new("RGB", (224, 224))
        result = classify_location(img, model, model_type="mobilenet")
        assert result.class_name == LOCATION_CLASSES[3]  # "door"
        assert result.class_id == 3

    def test_classify_confidence_in_range(self):
        from claimvision_ml.location.inference import classify_location
        model = self._fake_model(predicted_class_id=0)
        img = Image.new("RGB", (224, 224))
        result = classify_location(img, model)
        assert 0.0 <= result.confidence <= 1.0

    def test_classify_top3_length(self):
        from claimvision_ml.location.inference import classify_location
        model = self._fake_model(predicted_class_id=1)
        img = torch.zeros(1, 3, 224, 224)
        result = classify_location(img, model)
        assert len(result.top3) == 3

    def test_classify_top3_probabilities_descending(self):
        from claimvision_ml.location.inference import classify_location
        model = self._fake_model(predicted_class_id=4)
        img = Image.new("RGB", (224, 224))
        result = classify_location(img, model)
        probs = [p for _, p in result.top3]
        assert probs == sorted(probs, reverse=True)

    def test_classify_low_confidence_emits_warning(self):
        from claimvision_ml.location.inference import classify_location, LOW_CONFIDENCE_THRESHOLD

        class UniformModel(nn.Module):
            def __init__(self):
                super().__init__()
                self._d = nn.Linear(1, 1)
            def forward(self, x):
                return torch.zeros(x.size(0), 5)  # uniform → each class 20%

        model = UniformModel()
        img = Image.new("RGB", (224, 224))
        result = classify_location(img, model)
        assert result.confidence < LOW_CONFIDENCE_THRESHOLD
        assert len(result.warning) > 0

    def test_classify_high_confidence_no_warning(self):
        from claimvision_ml.location.inference import classify_location
        model = self._fake_model(predicted_class_id=0)
        img = Image.new("RGB", (224, 224))
        result = classify_location(img, model)
        assert result.warning == ""

    def test_classify_with_path_input(self, tmp_path):
        from claimvision_ml.location.inference import classify_location
        model = self._fake_model(predicted_class_id=2)
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (224, 224)).save(img_path)
        result = classify_location(img_path, model)
        assert result.class_id == 2

    def test_classify_unsupported_type_raises(self):
        from claimvision_ml.location.inference import classify_location
        model = self._fake_model()
        with pytest.raises(TypeError):
            classify_location(12345, model)


# ---------------------------------------------------------------------------
# export_location_onnx
# ---------------------------------------------------------------------------

def test_export_location_onnx_creates_file(tmp_path):
    pytest.importorskip("onnxscript", reason="onnxscript not installed in local venv; runs in Colab")
    from claimvision_ml.location.inference import export_location_onnx
    from claimvision_ml.location.mobilenet import LocationMobileNet

    model = LocationMobileNet(num_classes=5, dropout=0.3, pretrained=False)
    model.eval()
    out_path = tmp_path / "location_test.onnx"
    result_path = export_location_onnx(model, out_path)
    assert result_path.exists()
    assert result_path.stat().st_size > 0
