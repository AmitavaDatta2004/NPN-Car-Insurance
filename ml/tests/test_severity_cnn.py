"""Unit tests for ml/src/claimvision_ml/severity/cnn.py (Phase 5 — SEV-CNN-001).

All tests run WITHOUT a GPU and WITHOUT the real severity dataset.
They verify architecture, transforms, checkpoint round-trip, and utilities
so that the module can be validated on any machine immediately after pull.

Run:
    pytest ml/tests/test_severity_cnn.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Skip guard: if torch is not installed skip the whole module
# ─────────────────────────────────────────────────────────────────────────────
pytest.importorskip("torch", reason="PyTorch not installed — skipping severity CNN tests")


from claimvision_ml.severity.cnn import (  # noqa: E402
    CNN_IMAGE_SIZE,
    # new improvement: native 4:3 aspect ratio matching dataset median
    CNN_IMAGE_SIZE_4_3,
    SEBlock,
    SEVERITY_CLASS_TO_ID,
    SEVERITY_CLASSES,
    SEVERITY_ID_TO_CLASS,
    SeverityCNN,
    build_cnn_model,
    export_onnx,
    get_severity_transforms,
    load_cnn_model,
    measure_cpu_latency,
    predict_severity_cnn,
    save_cnn_checkpoint,
    save_cnn_preprocessing_config,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

class TestSeverityConstants:
    def test_classes_list(self):
        assert SEVERITY_CLASSES == ["minor", "moderate", "severe"]

    def test_class_to_id_mapping(self):
        assert SEVERITY_CLASS_TO_ID == {"minor": 0, "moderate": 1, "severe": 2}

    def test_id_to_class_mapping(self):
        assert SEVERITY_ID_TO_CLASS == {0: "minor", 1: "moderate", 2: "severe"}

    def test_cnn_image_size(self):
        assert CNN_IMAGE_SIZE == 160

    def test_cnn_image_size_4_3(self):
        # new improvement: native 4:3 aspect ratio matching dataset median (194x259)
        assert CNN_IMAGE_SIZE_4_3 == (192, 256)



# ─────────────────────────────────────────────────────────────────────────────
# SeverityCNN architecture
# ─────────────────────────────────────────────────────────────────────────────

class TestSeverityCNNArchitecture:
    def test_model_builds_with_defaults(self):
        model = SeverityCNN()
        assert isinstance(model, torch.nn.Module)

    def test_model_builds_without_extra_conv(self):
        model = SeverityCNN(use_extra_conv=False)
        assert isinstance(model, torch.nn.Module)

    def test_forward_pass_shape_default(self):
        model = SeverityCNN()
        model.eval()
        x = torch.zeros(2, 3, CNN_IMAGE_SIZE, CNN_IMAGE_SIZE)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 3), f"Expected (2, 3), got {logits.shape}"

    def test_forward_pass_shape_4_3(self):
        # new improvement: tests forward pass on native 4:3 (192, 256) full canvas
        model = SeverityCNN()
        model.eval()
        x = torch.zeros(2, 3, CNN_IMAGE_SIZE_4_3[0], CNN_IMAGE_SIZE_4_3[1])
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 3), f"Expected (2, 3), got {logits.shape}"

    def test_forward_pass_shape_no_extra_conv(self):
        model = SeverityCNN(use_extra_conv=False)
        model.eval()
        x = torch.zeros(2, 3, CNN_IMAGE_SIZE, CNN_IMAGE_SIZE)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 3)

    def test_output_is_finite(self):
        model = SeverityCNN()
        model.eval()
        x = torch.randn(1, 3, CNN_IMAGE_SIZE, CNN_IMAGE_SIZE)
        with torch.no_grad():
            logits = model(x)
        assert torch.isfinite(logits).all(), "Logits contain NaN or Inf"

    def test_parameter_count_returns_dict(self):
        model = SeverityCNN()
        counts = model.parameter_count()
        assert "trainable" in counts
        assert "total" in counts
        assert "frozen" in counts
        assert counts["trainable"] > 0
        assert counts["total"] == counts["trainable"] + counts["frozen"]

    def test_parameter_count_no_extra_conv_less_than_with(self):
        m_with = SeverityCNN(use_extra_conv=True).parameter_count()["total"]
        m_without = SeverityCNN(use_extra_conv=False).parameter_count()["total"]
        assert m_without < m_with

    def test_parameter_count_direct_head_less_than_two_layer(self):
        # new improvement: with extra_conv_channels=128, direct head saves 128*128 + 128 = 16,512 parameters
        m_direct = SeverityCNN(direct_head=True).parameter_count()["total"]
        m_two_layer = SeverityCNN(direct_head=False).parameter_count()["total"]
        assert m_direct < m_two_layer
        assert (m_two_layer - m_direct) == (128 * 128 + 128)

    def test_extra_conv_channels_parameter_count(self):
        # new improvement: extra_conv_channels=128 prevents the 295k parameter bottleneck of 128->256
        m_128 = SeverityCNN(extra_conv_channels=128).parameter_count()["total"]
        m_256 = SeverityCNN(extra_conv_channels=256).parameter_count()["total"]
        assert m_128 < m_256
        # Eliminating the 128->256 jump saves over 150k parameters
        assert (m_256 - m_128) > 150000


    def test_build_cnn_model_returns_severity_cnn(self):
        model = build_cnn_model()
        assert isinstance(model, SeverityCNN)

    def test_build_cnn_model_is_in_training_mode(self):
        model = build_cnn_model()
        assert model.training

    def test_default_classifier_is_two_layer(self):
        model = SeverityCNN()
        assert len(model.classifier) == 4
        assert isinstance(model.classifier[0], torch.nn.Linear)
        assert model.classifier[0].out_features == 128


# ─────────────────────────────────────────────────────────────────────────────
# Squeeze-and-Excitation (SEBlock)
# ─────────────────────────────────────────────────────────────────────────────

class TestSEBlock:
    def test_se_block_shape_preservation(self):
        se = SEBlock(channels=256, reduction=16)
        x = torch.randn(2, 256, 10, 10)
        out = se(x)
        assert out.shape == (2, 256, 10, 10)

    def test_se_block_scaling_factor_range(self):
        se = SEBlock(channels=64, reduction=16)
        x = torch.ones(1, 64, 5, 5)
        # Weight gate is sigmoid, output must be non-negative
        out = se(x)
        assert (out >= 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────

class TestSeverityTransforms:
    def test_train_transform_no_random_erasing(self):
        from torchvision import transforms
        t = get_severity_transforms("train")
        types = [type(op) for op in t.transforms]
        assert transforms.RandomErasing not in types

    def test_train_transform_returns_compose(self):
        from torchvision import transforms
        t = get_severity_transforms("train")
        assert isinstance(t, transforms.Compose)

    def test_val_transform_returns_compose(self):
        from torchvision import transforms
        t = get_severity_transforms("val")
        assert isinstance(t, transforms.Compose)

    def test_test_transform_returns_compose(self):
        from torchvision import transforms
        t = get_severity_transforms("test")
        assert isinstance(t, transforms.Compose)

    def test_invalid_split_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid split"):
            get_severity_transforms("unknown")  # type: ignore[arg-type]

    def test_train_transform_output_shape(self):
        # new improvement: default train transforms resize to full-canvas 4:3 (192, 256)
        from PIL import Image
        t = get_severity_transforms("train")
        img = Image.new("RGB", (300, 200))
        tensor = t(img)
        assert tensor.shape == (3, CNN_IMAGE_SIZE_4_3[0], CNN_IMAGE_SIZE_4_3[1])

    def test_val_transform_output_shape(self):
        # new improvement: default val transforms resize to full-canvas 4:3 (192, 256) without cropping
        from PIL import Image
        t = get_severity_transforms("val")
        img = Image.new("RGB", (300, 200))
        tensor = t(img)
        assert tensor.shape == (3, CNN_IMAGE_SIZE_4_3[0], CNN_IMAGE_SIZE_4_3[1])

    def test_legacy_square_image_size(self):
        # new improvement: backwards compatibility with legacy square size (160, 160)
        from PIL import Image
        t = get_severity_transforms("val", image_size=CNN_IMAGE_SIZE)
        img = Image.new("RGB", (300, 200))
        tensor = t(img)
        assert tensor.shape == (3, CNN_IMAGE_SIZE, CNN_IMAGE_SIZE)

    def test_custom_image_size(self):
        from PIL import Image
        t = get_severity_transforms("val", image_size=224)
        img = Image.new("RGB", (300, 200))
        tensor = t(img)
        assert tensor.shape == (3, 224, 224)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint save / load round-trip
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckpointRoundTrip:
    def test_save_and_load_roundtrip(self, tmp_path):
        model = SeverityCNN(dropout=0.3, use_extra_conv=True)
        ckpt_path = tmp_path / "severity_cnn_test.pt"
        save_cnn_checkpoint(model, ckpt_path, extra={"epoch": 1, "val_macro_f1": 0.7})

        loaded = load_cnn_model(ckpt_path, device="cpu")
        assert isinstance(loaded, SeverityCNN)
        assert loaded._dropout == 0.3
        assert loaded._use_extra_conv is True

        # Check that weights are identical
        for k in model.state_dict():
            assert torch.allclose(
                model.state_dict()[k].float(),
                loaded.state_dict()[k].float(),
            ), f"Mismatch in weight: {k}"

    def test_load_missing_checkpoint_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_cnn_model(tmp_path / "nonexistent.pt")


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing config
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessingConfig:
    def test_save_preprocessing_config_creates_valid_json(self, tmp_path):
        output = tmp_path / "cnn_preprocessing_config.json"
        save_cnn_preprocessing_config(output)
        assert output.exists()
        with open(output) as f:
            cfg = json.load(f)
        required_keys = {"model", "experiment_id", "image_size", "normalisation", "class_map"}
        assert required_keys.issubset(cfg.keys())

    def test_preprocessing_config_image_size(self, tmp_path):
        # new improvement: config defaults to native 4:3 image dimensions
        output = tmp_path / "cfg.json"
        save_cnn_preprocessing_config(output)
        with open(output) as f:
            cfg = json.load(f)
        assert cfg["image_size"] == [CNN_IMAGE_SIZE_4_3[0], CNN_IMAGE_SIZE_4_3[1]]

    def test_preprocessing_config_legacy_square_image_size(self, tmp_path):
        # new improvement: supports explicit square image size override
        output = tmp_path / "cfg_square.json"
        save_cnn_preprocessing_config(output, image_size=CNN_IMAGE_SIZE)
        with open(output) as f:
            cfg = json.load(f)
        assert cfg["image_size"] == [CNN_IMAGE_SIZE, CNN_IMAGE_SIZE]


    def test_preprocessing_config_class_map(self, tmp_path):
        output = tmp_path / "cfg.json"
        save_cnn_preprocessing_config(output)
        with open(output) as f:
            cfg = json.load(f)
        # Keys are strings in JSON
        assert cfg["class_map"]["0"] == "minor"
        assert cfg["class_map"]["1"] == "moderate"
        assert cfg["class_map"]["2"] == "severe"


# ─────────────────────────────────────────────────────────────────────────────
# ONNX export (skipped if onnx not installed)
# ─────────────────────────────────────────────────────────────────────────────

class TestOnnxExport:
    def test_onnx_export_creates_file(self, tmp_path):
        pytest.importorskip("onnx", reason="onnx not installed")
        model = SeverityCNN()
        model.eval()
        onnx_path = tmp_path / "severity_cnn.onnx"
        export_onnx(model, onnx_path)
        assert onnx_path.exists()
        assert onnx_path.stat().st_size > 0


# ─────────────────────────────────────────────────────────────────────────────
# CPU latency benchmark
# ─────────────────────────────────────────────────────────────────────────────

class TestLatencyBenchmark:
    def test_measure_cpu_latency_returns_dict(self):
        model = SeverityCNN()
        model.eval()
        result = measure_cpu_latency(model, n_warmup=2, n_runs=5)
        assert "mean_ms" in result
        assert "std_ms" in result
        assert "n_runs" in result
        assert result["n_runs"] == 5
        assert result["mean_ms"] > 0

    def test_latency_is_reasonable(self):
        """Sanity check: CPU forward pass should complete within 5000 ms."""
        model = SeverityCNN()
        model.eval()
        result = measure_cpu_latency(model, n_warmup=2, n_runs=3)
        assert result["mean_ms"] < 5000, (
            f"Unexpectedly slow: {result['mean_ms']} ms/image"
        )


# ─────────────────────────────────────────────────────────────────────────────
# predict_severity_cnn — requires a temporary image file
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictSeverityCNN:
    def test_predict_returns_expected_keys(self, tmp_path):
        from PIL import Image as PILImage

        # Create a tiny synthetic image
        img = PILImage.new("RGB", (200, 150), color=(128, 64, 32))
        img_path = tmp_path / "test_car.jpg"
        img.save(img_path)

        model = SeverityCNN()
        model.eval()
        result = predict_severity_cnn(str(img_path), model, device="cpu")

        expected_keys = {
            "predicted_class", "predicted_id", "probabilities",
            "confidence", "latency_ms", "model_version", "experiment_id",
        }
        assert expected_keys.issubset(result.keys())

    def test_predict_class_is_valid(self, tmp_path):
        from PIL import Image as PILImage

        img = PILImage.new("RGB", (200, 150))
        img_path = tmp_path / "test_car2.jpg"
        img.save(img_path)

        model = SeverityCNN()
        model.eval()
        result = predict_severity_cnn(str(img_path), model, device="cpu")

        assert result["predicted_class"] in SEVERITY_CLASSES
        assert result["predicted_id"] in (0, 1, 2)

    def test_predict_probabilities_sum_to_one(self, tmp_path):
        from PIL import Image as PILImage

        img = PILImage.new("RGB", (200, 150))
        img_path = tmp_path / "test_car3.jpg"
        img.save(img_path)

        model = SeverityCNN()
        model.eval()
        result = predict_severity_cnn(str(img_path), model, device="cpu")

        prob_sum = sum(result["probabilities"].values())
        assert abs(prob_sum - 1.0) < 1e-3, f"Probabilities sum to {prob_sum}, expected 1.0"

    def test_predict_model_version(self, tmp_path):
        from PIL import Image as PILImage

        img = PILImage.new("RGB", (200, 150))
        img_path = tmp_path / "test_car4.jpg"
        img.save(img_path)

        model = SeverityCNN()
        model.eval()
        result = predict_severity_cnn(str(img_path), model, device="cpu")
        assert result["model_version"] == "SEV-CNN-001"
        assert result["experiment_id"] == "SEV-CNN-001"

    def test_predict_with_calibrated_class_weights(self, tmp_path):
        from PIL import Image as PILImage

        img = PILImage.new("RGB", (200, 150))
        img_path = tmp_path / "test_car_weights.jpg"
        img.save(img_path)

        model = SeverityCNN()
        model.eval()
        # Biasing heavily towards moderate (class 1)
        result = predict_severity_cnn(
            str(img_path),
            model,
            device="cpu",
            class_weights=[0.01, 100.0, 0.01],
        )
        assert result["predicted_class"] == "moderate"
        assert result["predicted_id"] == 1
