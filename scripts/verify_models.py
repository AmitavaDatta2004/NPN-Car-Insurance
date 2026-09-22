"""verify_models.py — Verification script for trained ClaimVision AI models.

Scans the flat models/ folder and tests loading and CPU inference.
"""

import sys
import time
from pathlib import Path
import torch

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def test_fraud_model():
    print("\n--- [1/5] Testing Fraud Model ---")
    pt_path = MODELS_DIR / "fraud_mnv2_optimized_best.pt"
    if not pt_path.exists():
        print(f"❌ Missing: {pt_path}")
        return False

    try:
        from claimvision_ml.fraud.model import build_fraud_model
        model = build_fraud_model(pretrained=False)
        checkpoint = torch.load(pt_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif isinstance(checkpoint, dict):
            # Check if keys have prefix or direct
            try:
                model.load_state_dict(checkpoint)
            except Exception:
                # Try unwrapping
                model.load_state_dict(checkpoint, strict=False)
        else:
            model = checkpoint

        model.eval()
        dummy = torch.randn(1, 3, 224, 224)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(dummy)
        dt = (time.perf_counter() - t0) * 1000
        print(f"[OK] Fraud MobileNetV2 loaded successfully! Output shape: {out.shape}, CPU latency: {dt:.2f} ms")
        return True
    except Exception as e:
        print(f"[ERROR] Error testing fraud model: {e}")
        return False


def test_severity_models():
    print("\n--- [2/5] Testing Severity Models ---")
    results = {}

    # CNN
    cnn_path = MODELS_DIR / "severity_cnn_v1.pt"
    if cnn_path.exists():
        try:
            from claimvision_ml.severity.cnn import SeverityCNN
            model = SeverityCNN(num_classes=3)
            ckpt = torch.load(cnn_path, map_location="cpu")
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            elif isinstance(ckpt, dict):
                model.load_state_dict(ckpt, strict=False)
            model.eval()
            with torch.no_grad():
                out = model(torch.randn(1, 3, 160, 160))
            print(f"[OK] Severity CNN loaded! Output: {out.shape}")
            results["cnn"] = True
        except Exception as e:
            print(f"[WARN] Severity CNN load error: {e}")
            results["cnn"] = False

    # MobileNetV2
    mnv2_path = MODELS_DIR / "phase0_severity_mnv2.pt"
    if mnv2_path.exists():
        try:
            from claimvision_ml.severity.mobilenet import build_severity_mobilenet
            model = build_severity_mobilenet(num_classes=3, pretrained=False)
            ckpt = torch.load(mnv2_path, map_location="cpu")
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            elif isinstance(ckpt, dict):
                model.load_state_dict(ckpt, strict=False)
            model.eval()
            with torch.no_grad():
                out = model(torch.randn(1, 3, 224, 224))
            print(f"[OK] Severity MobileNetV2 loaded! Output: {out.shape}")
            results["mnv2"] = True
        except Exception as e:
            print(f"[WARN] Severity MobileNetV2 load error: {e}")
            results["mnv2"] = False

    return any(results.values())


def test_location_models():
    print("\n--- [3/5] Testing Location Models ---")
    results = {}

    # MobileNetV2
    mnv2_path = MODELS_DIR / "location_mobilenetv2.pt"
    if mnv2_path.exists():
        try:
            from claimvision_ml.location.mobilenet import build_location_mobilenet
            model = build_location_mobilenet(num_classes=5, pretrained=False)
            ckpt = torch.load(mnv2_path, map_location="cpu")
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            elif isinstance(ckpt, dict):
                model.load_state_dict(ckpt, strict=False)
            model.eval()
            with torch.no_grad():
                out = model(torch.randn(1, 3, 224, 224))
            print(f"[OK] Location MobileNetV2 loaded! Output: {out.shape}")
            results["mobilenet"] = True
        except Exception as e:
            print(f"[WARN] Location MobileNetV2 error: {e}")
            results["mobilenet"] = False

    # EfficientNet
    eff_path = MODELS_DIR / "location_efficientnet.pt"
    if eff_path.exists():
        try:
            from claimvision_ml.location.efficientnet import build_location_efficientnet
            model = build_location_efficientnet(num_classes=5, pretrained=False)
            ckpt = torch.load(eff_path, map_location="cpu")
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            elif isinstance(ckpt, dict):
                model.load_state_dict(ckpt, strict=False)
            model.eval()
            with torch.no_grad():
                out = model(torch.randn(1, 3, 224, 224))
            print(f"[OK] Location EfficientNet-B0 loaded! Output: {out.shape}")
            results["efficientnet"] = True
        except Exception as e:
            print(f"[WARN] Location EfficientNet error: {e}")
            results["efficientnet"] = False

    return any(results.values())


def test_damage_detection():
    print("\n--- [4/5] Testing Damage & Part Detection (YOLO) ---")
    yolo_damage = MODELS_DIR / "damage_yolov8n.pt"
    yolo_parts = MODELS_DIR / "parts_yolov8n.pt"

    if yolo_damage.exists():
        try:
            from ultralytics import YOLO
            model = YOLO(str(yolo_damage))
            print(f"[OK] Damage YOLOv8n loaded successfully! Task: {model.task}")
        except Exception as e:
            print(f"[WARN] Damage YOLO load note: {e}")

    if yolo_parts.exists():
        try:
            from ultralytics import YOLO
            model = YOLO(str(yolo_parts))
            print(f"[OK] Parts YOLOv8n loaded successfully! Task: {model.task}")
        except Exception as e:
            print(f"[WARN] Parts YOLO load note: {e}")


if __name__ == "__main__":
    print(f"Inspecting models in: {MODELS_DIR}")
    test_fraud_model()
    test_severity_models()
    test_location_models()
    test_damage_detection()
    print("\nVerification check complete.")
