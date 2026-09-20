"""Smoke tests for the claimvision_ml.fraud package.

All tests run on CPU without a real trained checkpoint. They verify:
- Model architecture builds and produces correct output shapes.
- Transform pipelines are valid and callable.
- FraudResult dataclass has all required fields.
- predict_fraud raises FileNotFoundError for missing inputs.
- Class weights have the correct ordering (suspicious > genuine weight).

Run with:
    pytest ml/tests/test_fraud_model.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from claimvision_ml.fraud import (
    FraudClassifier,
    FraudResult,
    build_fraud_model,
    get_class_weights,
    get_transforms,
    predict_fraud,
)


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------


class TestFraudClassifier:
    """Tests for FraudClassifier and build_fraud_model."""

    def test_build_fraud_model_pretrained_false(self):
        """Model builds without downloading weights (pretrained=False)."""
        model = build_fraud_model(pretrained=False)
        assert isinstance(model, FraudClassifier)

    def test_model_output_shape(self):
        """Forward pass with a random batch produces logit of shape (B, 1)."""
        model = build_fraud_model(pretrained=False)
        model.eval()
        x = torch.zeros(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 1), f"Expected (2, 1), got {out.shape}"

    def test_model_output_finite(self):
        """Logits must be finite (no NaN or inf)."""
        model = build_fraud_model(pretrained=False)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all(), "Model output contains NaN or inf"

    def test_freeze_backbone_no_backbone_grad(self):
        """After freeze_backbone, backbone parameters have requires_grad=False."""
        model = FraudClassifier(pretrained=False)
        model.freeze_backbone()
        for param in model.features.parameters():
            assert not param.requires_grad

    def test_unfreeze_final_blocks(self):
        """After unfreeze_final_blocks(2), last 2 feature blocks are trainable."""
        model = FraudClassifier(pretrained=False)
        model.freeze_backbone()
        model.unfreeze_final_blocks(n_blocks=2)
        total = len(model.features)
        for i, block in enumerate(model.features):
            if i >= total - 2:
                has_trainable = any(p.requires_grad for p in block.parameters())
                assert has_trainable, f"Block {i} should be trainable after unfreeze"

    def test_trainable_parameter_count_returns_dict(self):
        """trainable_parameter_count returns dict with required keys."""
        model = build_fraud_model(pretrained=False)
        counts = model.trainable_parameter_count()
        assert "trainable" in counts
        assert "total" in counts
        assert "frozen" in counts
        assert counts["trainable"] + counts["frozen"] == counts["total"]


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


class TestGetTransforms:
    """Tests for get_transforms."""

    def test_train_transform_returns_tensor(self):
        """Train transform converts a PIL image to a (3, 224, 224) tensor."""
        transform = get_transforms("train")
        img = Image.fromarray(_make_rgb_array())
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224), f"Expected (3,224,224), got {tensor.shape}"

    def test_val_transform_returns_tensor(self):
        """Val transform converts a PIL image to a (3, 224, 224) tensor."""
        transform = get_transforms("val")
        img = Image.fromarray(_make_rgb_array())
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_test_transform_same_as_val(self):
        """Test transform behaves identically to val transform."""
        import numpy as np
        rng = 42
        img = Image.fromarray(_make_rgb_array())
        t_val = get_transforms("val")
        t_test = get_transforms("test")
        # Both should produce the same output on the same deterministic image
        out_val = t_val(img)
        out_test = t_test(img)
        assert torch.allclose(out_val, out_test), "val and test transforms differ"

    def test_invalid_split_raises(self):
        """Passing an invalid split name should raise ValueError or similar."""
        with pytest.raises(Exception):
            get_transforms("unknown_split")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------


class TestGetClassWeights:
    """Tests for get_class_weights."""

    def test_suspicious_has_higher_weight(self, tmp_path):
        """pos_weight must be > 1 when genuine >> suspicious."""
        import pandas as pd
        # 100 genuine, 10 suspicious -> pos_weight = 10.0
        df = pd.DataFrame({
            "path": [f"img_{i}.jpg" for i in range(110)],
            "label": [0] * 100 + [1] * 10,
        })
        manifest = tmp_path / "manifest.csv"
        df.to_csv(manifest, index=False)

        weights = get_class_weights(str(manifest))
        assert weights.shape == (1,), "Expected shape (1,)"
        assert float(weights[0]) > 1.0, "pos_weight should be > 1 for imbalanced data"
        assert abs(float(weights[0]) - 10.0) < 0.01, "Expected pos_weight = 10.0"


# ---------------------------------------------------------------------------
# FraudResult dataclass
# ---------------------------------------------------------------------------


class TestFraudResult:
    """Tests for the FraudResult dataclass."""

    def test_fraud_result_has_required_fields(self):
        """FraudResult can be instantiated and has all required fields."""
        result = FraudResult(
            image_path="test.jpg",
            fraud_probability=0.42,
            risk_level="medium",
            route="CONTINUE",
            threshold_version="v1",
            model_version="FRAUD-MNV2-001",
            inference_ms=12.5,
            warnings=[],
        )
        assert result.image_path == "test.jpg"
        assert 0.0 <= result.fraud_probability <= 1.0
        assert result.risk_level in ("low", "medium", "high")
        assert result.route in ("CONTINUE", "FRAUD_REVIEW")
        assert isinstance(result.warnings, list)

    def test_fraud_result_default_warnings(self):
        """FraudResult warnings defaults to empty list."""
        result = FraudResult(
            image_path="x.jpg",
            fraud_probability=0.1,
            risk_level="low",
            route="CONTINUE",
            threshold_version="v1",
            model_version="FRAUD-MNV2-001",
            inference_ms=5.0,
        )
        assert result.warnings == []


# ---------------------------------------------------------------------------
# predict_fraud error handling
# ---------------------------------------------------------------------------


class TestPredictFraud:
    """Tests for predict_fraud error handling (no real checkpoint required)."""

    def test_missing_image_raises_file_not_found(self, tmp_path):
        """Raises FileNotFoundError when image does not exist."""
        with pytest.raises(FileNotFoundError, match="Image not found"):
            predict_fraud(
                image_path=tmp_path / "nonexistent.jpg",
                model_path=tmp_path / "model.pt",
                threshold_path=tmp_path / "thresholds.json",
            )

    def test_missing_model_raises_file_not_found(self, tmp_path):
        """Raises FileNotFoundError when model checkpoint does not exist."""
        # Create a dummy image so the image check passes
        img = Image.fromarray(_make_rgb_array())
        img_path = tmp_path / "real_image.jpg"
        img.save(str(img_path))

        with pytest.raises(FileNotFoundError):
            predict_fraud(
                image_path=img_path,
                model_path=tmp_path / "nonexistent_model.pt",
                threshold_path=tmp_path / "thresholds.json",
            )

    def test_missing_threshold_raises_file_not_found(self, tmp_path):
        """Raises FileNotFoundError when threshold file does not exist."""
        img = Image.fromarray(_make_rgb_array())
        img_path = tmp_path / "real_image.jpg"
        img.save(str(img_path))

        # Create a dummy checkpoint to pass the model check
        model = FraudClassifier(pretrained=False)
        ckpt_path = tmp_path / "model.pt"
        torch.save({"model_state_dict": model.state_dict(), "dropout": 0.3}, str(ckpt_path))

        with pytest.raises(FileNotFoundError, match="Threshold file not found"):
            predict_fraud(
                image_path=img_path,
                model_path=ckpt_path,
                threshold_path=tmp_path / "nonexistent_thresholds.json",
            )

    def test_full_inference_pipeline(self, tmp_path):
        """End-to-end smoke test with a dummy model and thresholds on a real image."""
        # Save dummy model checkpoint
        model = FraudClassifier(pretrained=False)
        model.eval()
        ckpt_path = tmp_path / "model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "dropout": 0.3,
                "model_version": "FRAUD-MNV2-TEST",
            },
            str(ckpt_path),
        )

        # Save dummy threshold JSON
        thresholds = {
            "version": "v_test",
            "model_version": "FRAUD-MNV2-TEST",
            "low_threshold": 0.3,
            "high_threshold": 0.6,
        }
        thr_path = tmp_path / "thresholds.json"
        thr_path.write_text(json.dumps(thresholds))

        # Create a real PIL image
        img = Image.fromarray(_make_rgb_array(size=300))
        img_path = tmp_path / "test_car.jpg"
        img.save(str(img_path))

        result = predict_fraud(
            image_path=img_path,
            model_path=ckpt_path,
            threshold_path=thr_path,
            device="cpu",
        )

        assert isinstance(result, FraudResult)
        assert 0.0 <= result.fraud_probability <= 1.0
        assert result.risk_level in ("low", "medium", "high")
        assert result.route in ("CONTINUE", "FRAUD_REVIEW")
        assert result.inference_ms > 0.0
        assert result.threshold_version == "v_test"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rgb_array(size: int = 400):
    """Create a simple numpy RGB array for testing."""
    import numpy as np
    return np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)


