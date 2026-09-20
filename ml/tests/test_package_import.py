"""Phase 0 smoke tests — verify the claimvision_ml package is importable.

These tests do not require any dataset, model weights, or heavy dependencies.
They confirm only that the package structure is correct and the version is set.
"""

import importlib

import pytest


def test_package_is_importable():
    """claimvision_ml must import without errors."""
    import claimvision_ml  # noqa: F401


def test_package_version():
    """Package version must be set and follow MAJOR.MINOR.PATCH format."""
    import claimvision_ml

    assert hasattr(claimvision_ml, "__version__"), "__version__ attribute missing"
    parts = claimvision_ml.__version__.split(".")
    assert len(parts) == 3, (
        f"Version must be MAJOR.MINOR.PATCH, got {claimvision_ml.__version__!r}"
    )
    assert claimvision_ml.__version__ == "0.1.0"


@pytest.mark.parametrize(
    "subpackage",
    [
        "claimvision_ml.data",
        "claimvision_ml.quality",
        "claimvision_ml.fraud",
        "claimvision_ml.severity",
        "claimvision_ml.detection",
        "claimvision_ml.costing",
        "claimvision_ml.pipeline",
    ],
)
def test_subpackage_is_importable(subpackage):
    """Every sub-package must import without errors."""
    mod = importlib.import_module(subpackage)
    assert mod is not None, f"{subpackage} returned None"
