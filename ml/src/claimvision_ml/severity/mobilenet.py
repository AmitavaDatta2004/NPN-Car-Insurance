"""MobileNetV2-based transfer-learning severity classifier for ClaimVision AI.

Architecture:
    224x224 RGB
    -> ImageNet-pretrained MobileNetV2 backbone (features)
    -> AdaptiveAvgPool2d((1, 1)) [1280-dim]
    -> Dropout(0.3)
    -> Linear(1280 -> 128) -> ReLU -> Dropout(0.3)
    -> Linear(128 -> 3) [logits: 0=minor, 1=moderate, 2=severe]

Experiment ID: SEV-MNV2-001
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from claimvision_ml.severity.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    SEVERITY_CLASS_TO_ID,
    SEVERITY_CLASSES,
    SEVERITY_ID_TO_CLASS,
)

logger = logging.getLogger(__name__)


class SeverityMobileNetV2(nn.Module):
    """MobileNetV2 backbone with a custom 3-class severity classification head.

    Supports two-stage transfer learning:
      - Stage A: freeze_backbone() to train head only.
      - Stage B: unfreeze_final_blocks(n_blocks=2) to fine-tune top layers with low LR.

    Args:
        pretrained: Whether to initialize backbone with ImageNet weights.
        num_classes: Number of target severity classes (default 3: minor, moderate, severe).
        dropout: Dropout probability in the classification head.
    """

    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)

        # 19 InvertedResidual blocks producing 1280 channels
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

        self.num_classes = num_classes
        self._dropout = dropout
        self._pretrained = pretrained

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw 3-class logits.

        Args:
            x: Input tensor of shape (B, 3, 224, 224).

        Returns:
            Logits of shape (B, 3).
        """
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train(self, mode: bool = True) -> SeverityMobileNetV2:
        """Set module in training mode, keeping frozen BatchNorm layers in eval mode."""
        super().train(mode)
        if mode:
            for module in self.features.modules():
                if isinstance(module, nn.BatchNorm2d):
                    if not any(p.requires_grad for p in module.parameters()):
                        module.eval()
        return self

    def freeze_backbone(self) -> None:
        """Freeze all parameters in the feature extractor (Stage A)."""
        for param in self.features.parameters():
            param.requires_grad = False
        for module in self.features.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def unfreeze_final_blocks(self, n_blocks: int = 2) -> None:
        """Unfreeze the last n_blocks InvertedResidual layers for fine-tuning (Stage B).

        MobileNetV2 has 19 feature blocks (0 to 18).
        n_blocks=2 unfreezes features[17] and features[18].
        """
        total = len(self.features)
        for i, block in enumerate(self.features):
            if i >= total - n_blocks:
                for param in block.parameters():
                    param.requires_grad = True

    def trainable_parameter_count(self) -> dict[str, int]:
        """Return counts of trainable, frozen, and total parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {"trainable": trainable, "frozen": total - trainable, "total": total}


def build_severity_mobilenet(
    pretrained: bool = True,
    num_classes: int = 3,
    dropout: float = 0.3,
) -> SeverityMobileNetV2:
    """Factory function to instantiate SeverityMobileNetV2."""
    return SeverityMobileNetV2(
        pretrained=pretrained,
        num_classes=num_classes,
        dropout=dropout,
    )


def save_severity_checkpoint(
    model: SeverityMobileNetV2,
    save_path: str | Path | None = None,
    epoch: int = 0,
    metrics: dict[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    extra_meta: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Path:
    """Save model weights and metadata to a PyTorch .pt checkpoint."""
    target_path = save_path or kwargs.get("path")
    if target_path is None:
        raise ValueError("Must provide save_path or path to save_severity_checkpoint")
    path = Path(target_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert metrics to plain Python types
    clean_metrics: dict[str, Any] = {}
    for k, v in (metrics or {}).items():
        if isinstance(v, (float, int, str, bool)):
            clean_metrics[k] = v
        elif hasattr(v, "tolist"):
            clean_metrics[k] = v.tolist()
        elif hasattr(v, "item"):
            clean_metrics[k] = v.item()
        elif isinstance(v, (list, dict)):
            clean_metrics[k] = v

    for k in ["val_loss", "val_macro_f1", "val_weighted_f1"]:
        if k in kwargs and k not in clean_metrics:
            val = kwargs[k]
            clean_metrics[k] = val.item() if hasattr(val, "item") else val

    extra = extra_meta or kwargs.get("extra") or {}

    checkpoint = {
        "model_architecture": "mobilenet_v2",
        "experiment_id": "SEV-MNV2-001",
        "num_classes": model.num_classes,
        "classes": list(SEVERITY_CLASSES),
        "class_to_id": SEVERITY_CLASS_TO_ID,
        "id_to_class": SEVERITY_ID_TO_CLASS,
        "image_size": [224, 224],
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "epoch": epoch,
        "metrics": clean_metrics,
        "state_dict": model.state_dict(),
        "extra_meta": extra,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, str(path))
    logger.info(f"Saved severity checkpoint to {path}")
    return path


def load_severity_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[SeverityMobileNetV2, dict[str, Any]]:
    """Load a SeverityMobileNetV2 model from a .pt checkpoint.

    Returns:
        (model in eval mode, metadata dict)
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    try:
        chk = torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        chk = torch.load(str(path), map_location=device)

    num_classes = chk.get("num_classes", 3)

    model = SeverityMobileNetV2(pretrained=False, num_classes=num_classes)
    state_dict = chk.get("state_dict") or chk.get("model_state_dict")
    if state_dict is None:
        raise KeyError("No 'state_dict' or 'model_state_dict' found in checkpoint.")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    meta = {k: v for k, v in chk.items() if k not in ("state_dict", "model_state_dict")}
    return model, meta


def export_onnx(
    model: SeverityMobileNetV2,
    save_path: str | Path,
    input_size: tuple[int, int, int, int] = (1, 3, 224, 224),
    dynamic_batch: bool = True,
    verify: bool = True,
) -> Path:
    """Export trained model to ONNX format with verification.

    Args:
        model: SeverityMobileNetV2 model.
        save_path: Output .onnx path.
        input_size: Dummy input tensor shape (B, C, H, W).
        dynamic_batch: Allow dynamic batch sizes on dimension 0.
        verify: Run inference parity check between PyTorch and ONNX.

    Returns:
        Path to exported ONNX file.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_size, device=device)

    dynamic_axes = {"input": {0: "batch_size"}, "logits": {0: "batch_size"}} if dynamic_batch else None

    model.eval()
    with torch.no_grad():
        try:
            torch.onnx.export(
                model,
                dummy_input,
                str(path),
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["logits"],
                dynamic_axes=dynamic_axes,
                dynamo=False,
            )
        except TypeError:
            torch.onnx.export(
                model,
                dummy_input,
                str(path),
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["logits"],
                dynamic_axes=dynamic_axes,
            )

    logger.info(f"ONNX model exported to {path}")

    if verify:
        try:
            import onnxruntime as ort

            ort_session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            with torch.no_grad():
                torch_out = model(dummy_input.cpu()).numpy()

            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
            ort_out = ort_session.run(None, ort_inputs)[0]

            diff = float(abs(torch_out - ort_out).max())
            if diff > 1e-3:
                logger.warning(f"ONNX parity difference {diff:.6f} exceeds threshold")
            else:
                logger.info(f"ONNX parity verified: max diff = {diff:.6e}")
        except ImportError:
            logger.info("onnxruntime not installed; skipping ONNX runtime verification")

    return path


def audit_batchnorm(model: nn.Module) -> dict[str, list[str]]:
    """Audit BatchNorm layers in a model to verify frozen vs trainable states.

    Identifies whether frozen BatchNorm layers are in train or eval mode.

    Args:
        model: PyTorch neural network module.

    Returns:
        Dictionary with lists of trainable_bn, frozen_bn_train, and frozen_bn_eval layer names.
    """
    frozen_bn_train: list[str] = []
    frozen_bn_eval: list[str] = []
    trainable_bn: list[str] = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue

        params = list(module.parameters(recurse=False))
        is_trainable = any(p.requires_grad for p in params)

        if is_trainable:
            trainable_bn.append(name)
        elif module.training:
            frozen_bn_train.append(name)
        else:
            frozen_bn_eval.append(name)

    return {
        "trainable_bn": trainable_bn,
        "frozen_bn_train": frozen_bn_train,
        "frozen_bn_eval": frozen_bn_eval,
    }


def set_frozen_batchnorm_eval(model: nn.Module) -> None:
    """Force all frozen BatchNorm layers into eval mode to preserve pretrained statistics."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            params = list(module.parameters(recurse=False))
            if params and not any(p.requires_grad for p in params):
                module.eval()


def make_class_weights(
    values: list[float] | tuple[float, ...] | torch.Tensor,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Construct mean-normalized class weights tensor for severity classification."""
    weights = torch.tensor(values, dtype=torch.float32, device=device)
    return weights / weights.mean()


class HybridOrdinalLoss(nn.Module):
    """Hybrid loss combining cross-entropy and smooth L1 ordinal distance penalty.

    Total Loss = CE(logits, labels; class_weights, label_smoothing) + lambda_ordinal * SmoothL1(E[severity], labels)

    Args:
        class_weights: Tensor of shape (num_classes,) containing class weights.
        label_smoothing: Cross-entropy label smoothing factor (default 0.05).
        lambda_ordinal: Weight for the smooth L1 ordinal distance penalty (default 0.2).
    """

    def __init__(
        self,
        class_weights: torch.Tensor,
        label_smoothing: float = 0.05,
        lambda_ordinal: float = 0.2,
    ) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights.clone().detach())
        self.label_smoothing = label_smoothing
        self.lambda_ordinal = lambda_ordinal
        self.register_buffer("severity_values", torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32))

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute hybrid loss and individual components.

        Args:
            logits: Predicted class logits (B, 3).
            labels: Ground truth class targets (B,).

        Returns:
            Tuple of (total_loss, ce_loss, ordinal_loss).
        """
        device = logits.device
        dtype = logits.dtype
        class_weights = self.class_weights.to(device=device, dtype=dtype)
        severity_values = self.severity_values.to(device=device, dtype=dtype)

        ce_loss = F.cross_entropy(
            logits,
            labels,
            weight=class_weights,
            label_smoothing=self.label_smoothing,
        )

        probabilities = torch.softmax(logits, dim=1)
        predicted_severity = (probabilities * severity_values.unsqueeze(0)).sum(dim=1)
        ordinal_loss = F.smooth_l1_loss(predicted_severity, labels.to(device=device, dtype=dtype))
        total_loss = ce_loss + self.lambda_ordinal * ordinal_loss

        return total_loss, ce_loss, ordinal_loss


def compute_ordinal_error_metrics(
    predictions: list[int] | np.ndarray, labels: list[int] | np.ndarray
) -> dict[str, float | int]:
    """Compute ordinal-specific metrics including MAE, extreme errors, and adjacent errors.

    Classes: 0=Minor, 1=Moderate, 2=Severe.
    Adjacent errors: |y_pred - y_true| == 1 (minor <-> moderate, moderate <-> severe)
    Extreme errors: |y_pred - y_true| == 2 (minor <-> severe)

    Returns:
        dict with keys:
          - severity_mae: Mean Absolute Error between predicted and true severity indices
          - extreme_error_rate: Fraction of predictions with distance == 2
          - extreme_errors: Absolute count of minor <-> severe mistakes
          - adjacent_errors: Absolute count of distance == 1 mistakes
          - minor_mod_errors: Count of minor predicted as moderate or vice versa
          - mod_sev_errors: Count of moderate predicted as severe or vice versa
    """
    preds = np.asarray(predictions)
    targets = np.asarray(labels)
    diffs = np.abs(preds - targets)

    severity_mae = float(np.mean(diffs))
    extreme_error_rate = float(np.mean(diffs == 2))
    extreme_errors = int(np.sum(diffs == 2))
    adjacent_errors = int(np.sum(diffs == 1))

    minor_mod_errors = int(np.sum(((preds == 0) & (targets == 1)) | ((preds == 1) & (targets == 0))))
    mod_sev_errors = int(np.sum(((preds == 1) & (targets == 2)) | ((preds == 2) & (targets == 1))))

    return {
        "severity_mae": severity_mae,
        "extreme_error_rate": extreme_error_rate,
        "extreme_errors": extreme_errors,
        "adjacent_errors": adjacent_errors,
        "minor_mod_errors": minor_mod_errors,
        "mod_sev_errors": mod_sev_errors,
    }
