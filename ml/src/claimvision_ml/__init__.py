"""ClaimVision ML — machine-learning library for ClaimVision AI.

Sub-packages
------------
data        Dataset loading, manifests, and split utilities.
quality     OpenCV-based image-quality and evidence-integrity checks.
fraud       Visual fraud-risk classifier (MobileNetV2) and signals.
severity    Severity classifiers: baseline CNN, MobileNetV2, ViT-Tiny.
detection   YOLO-based damage localisation and damaged-part detection.
costing     Rule-based repair-cost estimation engine.
pipeline    Unified inference orchestrator (assess_claim).
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
