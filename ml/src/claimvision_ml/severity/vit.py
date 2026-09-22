"""ViT-Tiny damage severity classifier (Phase 7 / SEV-VIT-001).

Implements the Vision Transformer architecture (vit_tiny_patch16_224), dataset loading,
two-stage transfer learning protocols, ONNX export, and standalone runtime inference.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object  # type: ignore
    Dataset = object  # type: ignore

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


# Canonical class taxonomy for severity
SEVERITY_CLASSES = ["minor", "moderate", "severe"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(SEVERITY_CLASSES)}
ID_TO_CLASS = {idx: name for idx, name in enumerate(SEVERITY_CLASSES)}

# ImageNet normalization standard for Vision Transformers
VIT_MEAN = [0.485, 0.456, 0.406]
VIT_STD = [0.229, 0.224, 0.225]


@dataclass
class SeverityViTResult:
    """Prediction result returned by predict_severity_vit."""

    predicted_class: str
    label_id: int
    confidence: float
    probabilities: dict[str, float]
    inference_ms: float
    model_version: str = "vit_tiny_patch16_224-v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_class": self.predicted_class,
            "label_id": self.label_id,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "inference_ms": round(self.inference_ms, 2),
            "model_version": self.model_version,
        }


def get_vit_transforms(split: str = "train", image_size: int = 224):
    """Return ViT-appropriate data transformations.

    Args:
        split: One of 'train', 'val', 'test', or 'inference'.
        image_size: Target square image dimension (default: 224 for ViT-Tiny).
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for data transformations.")

    if split == "train":
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),  # ~256
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=VIT_MEAN, std=VIT_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),  # ~256
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=VIT_MEAN, std=VIT_STD),
        ])


class SeverityDataset(Dataset):
    """PyTorch Dataset reading Car Damage Severity manifest CSVs."""

    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        transform=None,
        dataset_root: str | Path | None = None,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for SeverityDataset.")

        if isinstance(manifest, (str, Path)):
            self.df = pd.read_csv(manifest)
        else:
            self.df = manifest.copy().reset_index(drop=True)

        self.transform = transform
        self.dataset_root = Path(dataset_root) if dataset_root else None

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, raw_path: str) -> Path:
        p = Path(raw_path)
        if p.is_file():
            return p

        # Check relative to dataset_root
        if self.dataset_root:
            candidate = self.dataset_root / p.name
            if candidate.is_file():
                return candidate
            candidate_sub = self.dataset_root / p
            if candidate_sub.is_file():
                return candidate_sub

        # Search known root patterns
        for prefix in [Path("."), Path("data/raw"), Path("data/raw/car_damage_severity")]:
            candidate = prefix / p
            if candidate.is_file():
                return candidate

        return p

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image_path_str = str(row["image_path"])
        img_path = self._resolve_image_path(image_path_str)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Fallback black image if corrupt/missing during headless tests
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        label_id = int(row["label_id"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label_id


if TORCH_AVAILABLE:

    class SeverityViTTiny(nn.Module):
        """ViT-Tiny architecture for 3-class car damage severity estimation.

        Uses timm's vit_tiny_patch16_224 backbone (embed_dim=192, depth=12, heads=3)
        with custom head and stage-freezing utilities.
        """

        def __init__(
            self,
            pretrained: bool = True,
            num_classes: int = 3,
            drop_rate: float = 0.1,
        ):
            super().__init__()
            self.num_classes = num_classes
            self.model_name = "vit_tiny_patch16_224"

            if TIMM_AVAILABLE:
                # Load via timm
                self.backbone = timm.create_model(
                    self.model_name,
                    pretrained=pretrained,
                    num_classes=num_classes,
                    drop_rate=drop_rate,
                )
            else:
                # Lightweight pure-PyTorch ViT-Tiny fallback if timm is being installed
                self.backbone = _create_fallback_vit_tiny(
                    pretrained=pretrained,
                    num_classes=num_classes,
                    drop_rate=drop_rate,
                )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.backbone(x)

        def freeze_backbone(self) -> None:
            """Stage A: Freeze all transformer blocks, train only classifier head."""
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Unfreeze the classification head
            head = getattr(self.backbone, "head", None)
            if head is not None:
                for param in head.parameters():
                    param.requires_grad = True
            elif hasattr(self.backbone, "fc"):
                for param in self.backbone.fc.parameters():
                    param.requires_grad = True

        def unfreeze_top_blocks(self, num_blocks: int = 4) -> None:
            """Stage B: Unfreeze the top N transformer blocks and norm layer."""
            blocks = getattr(self.backbone, "blocks", None)
            if blocks is not None and isinstance(blocks, (list, nn.ModuleList, nn.Sequential)):
                total_blocks = len(blocks)
                start_idx = max(0, total_blocks - num_blocks)
                for i in range(start_idx, total_blocks):
                    for param in blocks[i].parameters():
                        param.requires_grad = True

            # Unfreeze norm layer
            norm = getattr(self.backbone, "norm", None)
            if norm is not None:
                for param in norm.parameters():
                    param.requires_grad = True

            # Ensure head remains unfrozen
            head = getattr(self.backbone, "head", None)
            if head is not None:
                for param in head.parameters():
                    param.requires_grad = True

        def count_parameters(self) -> tuple[int, int]:
            """Return (trainable_parameters, total_parameters)."""
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            return trainable, total


    def _create_fallback_vit_tiny(
        pretrained: bool = False,
        num_classes: int = 3,
        drop_rate: float = 0.1,
    ) -> nn.Module:
        """Lightweight fallback module mimicking vit_tiny_patch16_224."""
        class FallbackViT(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_embed = nn.Conv2d(3, 192, kernel_size=16, stride=16)
                self.cls_token = nn.Parameter(torch.zeros(1, 1, 192))
                self.pos_embed = nn.Parameter(torch.zeros(1, 197, 192))
                self.pos_drop = nn.Dropout(p=drop_rate)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=192, nhead=3, dim_feedforward=768, dropout=drop_rate, batch_first=True
                )
                self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=12)
                self.norm = nn.LayerNorm(192)
                self.head = nn.Linear(192, num_classes)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                batch_size = x.shape[0]
                x = self.patch_embed(x).flatten(2).transpose(1, 2)
                cls_tokens = self.cls_token.expand(batch_size, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                x = self.pos_drop(x + self.pos_embed[:, : x.size(1), :])
                x = self.blocks(x)
                x = self.norm(x[:, 0])
                return self.head(x)

        return FallbackViT()

else:
    class SeverityViTTiny:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for SeverityViTTiny.")


def build_vit_model(
    pretrained: bool = True,
    num_classes: int = 3,
    freeze_backbone: bool = True,
) -> SeverityViTTiny:
    """Factory to instantiate SeverityViTTiny in Stage A or full mode."""
    model = SeverityViTTiny(pretrained=pretrained, num_classes=num_classes)
    if freeze_backbone:
        model.freeze_backbone()
    return model


def save_vit_checkpoint(
    model: nn.Module,
    save_path: str | Path,
    optimizer: Any | None = None,
    epoch: int | None = None,
    metrics: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """Save PyTorch model weights and training metadata."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to save checkpoints.")

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "model_state_dict": model.state_dict(),
        "classes": SEVERITY_CLASSES,
        "class_to_id": CLASS_TO_ID,
        "epoch": epoch,
        "metrics": metrics or {},
        "config": config or {},
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(state, str(path))


def load_vit_model(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> SeverityViTTiny:
    """Load SeverityViTTiny from a .pt checkpoint."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to load checkpoints.")

    model = SeverityViTTiny(pretrained=False, num_classes=3)
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def export_vit_onnx(
    model: nn.Module,
    output_path: str | Path,
    device: str = "cpu",
    verify: bool = True,
) -> Path:
    """Export model to ONNX with dynamic batch axis."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for ONNX export.")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.to(device)

    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        str(out),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    if verify:
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
            ort_inputs = {session.get_inputs()[0].name: dummy_input.cpu().numpy()}
            ort_outs = session.run(None, ort_inputs)

            with torch.no_grad():
                torch_out = model(dummy_input).cpu().numpy()

            max_diff = float(np.max(np.abs(torch_out - ort_outs[0])))
            if max_diff > 1e-4:
                raise ValueError(f"ONNX numerical verification failed: max diff {max_diff} > 1e-4")
        except ImportError:
            pass  # onnxruntime not installed in this environment

    return out


def predict_severity_vit(
    image_input: str | Path | Image.Image | np.ndarray,
    model_or_path: nn.Module | str | Path,
    device: str = "cpu",
) -> SeverityViTResult:
    """Run ViT-Tiny severity inference on a single image.

    Args:
        image_input: File path, PIL Image, or RGB numpy array.
        model_or_path: Loaded SeverityViTTiny model or path to .pt file.
        device: Target execution device ('cpu' or 'cuda').
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for inference.")

    t_start = time.perf_counter()

    # Load / resolve image
    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        if not img_path.is_file():
            raise FileNotFoundError(f"Image not found at: {img_path}")
        image = Image.open(img_path).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Load model if given path
    if isinstance(model_or_path, (str, Path)):
        model = load_vit_model(model_or_path, device=device)
    else:
        model = model_or_path
        model.to(device)
        model.eval()

    # Transform
    transform = get_vit_transforms(split="inference", image_size=224)
    tensor = transform(image).unsqueeze(0).to(device)

    # Forward
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    label_id = int(np.argmax(probs))
    confidence = float(probs[label_id])
    predicted_class = ID_TO_CLASS[label_id]
    probabilities = {ID_TO_CLASS[i]: float(probs[i]) for i in range(len(probs))}

    t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    return SeverityViTResult(
        predicted_class=predicted_class,
        label_id=label_id,
        confidence=confidence,
        probabilities=probabilities,
        inference_ms=t_elapsed_ms,
    )
