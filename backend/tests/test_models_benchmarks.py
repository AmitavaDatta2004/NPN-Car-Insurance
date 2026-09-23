"""Unit and integration tests for /api/v1/models endpoints.

Verifies exact ground-truth benchmark metrics, confusion matrices,
and scikit-learn classification reports from training notebooks.
"""

import os
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.main import app  # noqa: E402

client = TestClient(app)


def test_models_benchmark_endpoint_status_and_schema():
    response = client.get("/api/v1/models/benchmark")
    assert response.status_code == 200
    data = response.json()

    assert "severity_models" in data
    assert "location_models" in data
    assert "fraud_models" in data
    assert "detection_models" in data
    assert "active_models" in data

    assert len(data["severity_models"]) == 3
    assert len(data["location_models"]) == 2
    assert len(data["fraud_models"]) == 2
    assert len(data["detection_models"]) == 2


def test_severity_models_ground_truth_metrics():
    response = client.get("/api/v1/models/benchmark")
    data = response.json()
    sev_map = {m["id"]: m for m in data["severity_models"]}

    # MobileNetV2 (Winner)
    mnv2 = sev_map["mobilenet_v2"]
    assert mnv2["is_winner"] is True
    assert mnv2["accuracy"] == 0.6774
    assert mnv2["macro_f1"] == 0.6777
    assert mnv2["confusion_matrix"] == [
        [62, 17, 3],
        [18, 46, 17],
        [3, 22, 60],
    ]
    report_mnv2 = mnv2["classification_report"]
    assert report_mnv2["total_support"] == 248
    assert report_mnv2["classes"]["minor"]["support"] == 82
    assert report_mnv2["classes"]["moderate"]["support"] == 81
    assert report_mnv2["classes"]["severe"]["support"] == 85

    # Baseline CNN
    cnn = sev_map["baseline_cnn"]
    assert cnn["is_winner"] is False
    assert cnn["accuracy"] == 0.6048
    assert cnn["macro_f1"] == 0.6055
    assert cnn["confusion_matrix"] == [
        [58, 16, 8],
        [15, 42, 24],
        [8, 27, 50],
    ]

    # ViT-Tiny
    vit = sev_map["vit_tiny"]
    assert vit["is_winner"] is False
    assert vit["accuracy"] == 0.6210
    assert vit["macro_f1"] == 0.6119
    assert vit["confusion_matrix"] == [
        [67, 9, 6],
        [31, 35, 15],
        [12, 21, 52],
    ]


def test_location_and_fraud_models_ground_truth_metrics():
    response = client.get("/api/v1/models/benchmark")
    data = response.json()
    loc_map = {m["id"]: m for m in data["location_models"]}

    # EfficientNet-B0 (Winner)
    eff = loc_map["efficientnet_b0"]
    assert eff["is_winner"] is True
    assert eff["accuracy"] == 0.7500
    assert eff["macro_f1"] == 0.6914
    assert eff["confusion_matrix"] == [
        [3, 0, 1, 0, 0],
        [0, 2, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0],
        [0, 0, 1, 0, 2],
    ]

    # Fraud Winner (Optimized MobileNetV2)
    fraud_map = {m["id"]: m for m in data["fraud_models"]}
    fraud_opt = fraud_map["fraud_mnv2_opt"]
    assert fraud_opt["is_winner"] is True
    assert fraud_opt["accuracy"] == 0.9646
    assert fraud_opt["confusion_matrix"] == [
        [1115, 28],
        [15, 56],
    ]


def test_active_model_toggle():
    payload = {"task": "severity", "model_id": "baseline_cnn"}
    post_res = client.post("/api/v1/models/active", json=payload)
    assert post_res.status_code == 200
    assert post_res.json()["active_model"] == "baseline_cnn"

    # Reset back to winning mobilenet_v2
    client.post("/api/v1/models/active", json={"task": "severity", "model_id": "mobilenet_v2"})
