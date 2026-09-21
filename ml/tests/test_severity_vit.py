"""Unit tests for the ViT-Tiny severity classification module (SEV-VIT-001).

Covers architecture instantiation, tensor forward pass, Stage A/B layer freezing,
dataset and transform pipelines, checkpoint serialization, and prediction dataclass.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from claimvision_ml.severity.vit import (
    CLASS_TO_ID,
    ID_TO_CLASS,
    SEVERITY_CLASSES,
    SeverityDataset,
    SeverityViTResult,
    SeverityViTTiny,
    build_vit_model,
    get_vit_transforms,
    load_vit_model,
    predict_severity_vit,
    save_vit_checkpoint,
)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def test_severity_classes_taxonomy():
    """Verify 3-class canonical severity taxonomy and bidirectional mappings."""
    assert SEVERITY_CLASSES == ["minor", "moderate", "severe"]
    assert CLASS_TO_ID["minor"] == 0
    assert CLASS_TO_ID["moderate"] == 1
    assert CLASS_TO_ID["severe"] == 2
    assert ID_TO_CLASS[0] == "minor"
    assert ID_TO_CLASS[1] == "moderate"
    assert ID_TO_CLASS[2] == "severe"


def test_vit_result_dataclass():
    """Verify SeverityViTResult dataclass serialization and field values."""
    res = SeverityViTResult(
        predicted_class="moderate",
        label_id=1,
        confidence=0.8765,
        probabilities={"minor": 0.05, "moderate": 0.8765, "severe": 0.0735},
        inference_ms=18.42,
    )
    d = res.to_dict()
    assert d["predicted_class"] == "moderate"
    assert d["label_id"] == 1
    assert d["confidence"] == 0.8765
    assert d["inference_ms"] == 18.42
    assert d["model_version"] == "vit_tiny_patch16_224-v1"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required for tensor tests")
def test_vit_transforms_shapes():
    """Verify ViT transforms produce correct (3, 224, 224) normalized tensors."""
    img = Image.new("RGB", (320, 240), color=(120, 80, 50))

    train_tf = get_vit_transforms(split="train", image_size=224)
    eval_tf = get_vit_transforms(split="val", image_size=224)

    t_train = train_tf(img)
    t_eval = eval_tf(img)

    assert isinstance(t_train, torch.Tensor)
    assert t_train.shape == (3, 224, 224)
    assert isinstance(t_eval, torch.Tensor)
    assert t_eval.shape == (3, 224, 224)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required for dataset tests")
def test_severity_dataset():
    """Verify SeverityDataset correctly loads and formats samples from a DataFrame."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_img = Path(tmpdir) / "test_img.jpg"
        Image.new("RGB", (200, 200), color=(50, 100, 150)).save(tmp_img)

        df = pd.DataFrame([
            {"image_path": str(tmp_img), "label": "severe", "label_id": 2},
        ])

        tf = get_vit_transforms(split="val", image_size=224)
        ds = SeverityDataset(manifest=df, transform=tf)

        assert len(ds) == 1
        tensor, label_id = ds[0]
        assert tensor.shape == (3, 224, 224)
        assert label_id == 2


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required for model tests")
def test_vit_model_instantiation_and_forward():
    """Verify SeverityViTTiny builds, produces logits (B, 3), and reports parameter counts."""
    model = SeverityViTTiny(pretrained=False, num_classes=3)
    dummy = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        logits = model(dummy)

    assert logits.shape == (2, 3)
    trainable, total = model.count_parameters()
    assert total > 1_000_000  # ViT-Tiny is ~5.7M parameters
    assert trainable == total


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required for freezing tests")
def test_vit_stage_a_and_stage_b_freezing():
    """Verify Stage A freezes backbone and Stage B unfreezes specified transformer blocks."""
    model = build_vit_model(pretrained=False, num_classes=3, freeze_backbone=True)
    trainable_a, total = model.count_parameters()

    # In Stage A, only classification head is trainable (much fewer than total)
    assert trainable_a < total
    assert trainable_a > 0

    # In Stage B, unfreeze top 4 blocks
    model.unfreeze_top_blocks(num_blocks=4)
    trainable_b, _ = model.count_parameters()

    assert trainable_b > trainable_a
    assert trainable_b <= total


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required for serialization tests")
def test_save_and_load_vit_checkpoint():
    """Verify saving and loading weights reproduces identical predictions."""
    model = SeverityViTTiny(pretrained=False, num_classes=3)
    dummy = torch.randn(1, 3, 224, 224)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_vit.pt"
        save_vit_checkpoint(model, ckpt_path, epoch=1, metrics={"val_macro_f1": 0.75})

        assert ckpt_path.is_file()

        model.eval()
        loaded_model = load_vit_model(ckpt_path)
        with torch.no_grad():
            orig_out = model(dummy)
            loaded_out = loaded_model(dummy)

        diff = torch.max(torch.abs(orig_out - loaded_out)).item()
        assert diff < 1e-6


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required for inference tests")
def test_predict_severity_vit():
    """Verify predict_severity_vit generates valid probabilities and prediction result."""
    model = SeverityViTTiny(pretrained=False, num_classes=3)

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "sample.jpg"
        Image.new("RGB", (250, 250), color=(180, 50, 50)).save(img_path)

        res = predict_severity_vit(img_path, model)

        assert res.predicted_class in SEVERITY_CLASSES
        assert 0 <= res.label_id <= 2
        assert 0.0 <= res.confidence <= 1.0
        assert len(res.probabilities) == 3
        assert sum(res.probabilities.values()) == pytest.approx(1.0, rel=1e-3)
        assert res.inference_ms > 0.0
