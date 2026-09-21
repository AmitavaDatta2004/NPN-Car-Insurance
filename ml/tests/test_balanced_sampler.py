"""Smoke tests for BalancedEpochSampler (ML-003).

Tests verify:
1. Epoch size is correct for 50:50 ratio.
2. ALL suspicious indices appear in every epoch — fixed, no randomness.
3. Genuine indices are DIFFERENT across epochs — fresh random sample each time.
4. All four target ratios (50:50, 40:60, 30:70, 20:80) produce correct epoch sizes.

These tests use a synthetic in-memory dataset so no disk I/O is needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from claimvision_ml.fraud.dataset import BalancedEpochSampler, FraudDataset


# ---------------------------------------------------------------------------
# Helpers — build a minimal synthetic FraudDataset without touching disk
# ---------------------------------------------------------------------------

class _FakeFraudDataset:
    """Minimal stand-in for FraudDataset for sampler tests (no image loading)."""

    def __init__(self, n_genuine: int = 200, n_suspicious: int = 50) -> None:
        paths = [f"/fake/{i}.jpg" for i in range(n_genuine + n_suspicious)]
        labels = [0] * n_genuine + [1] * n_suspicious
        self.df = pd.DataFrame({"path": paths, "label": labels})

    def __len__(self) -> int:
        return len(self.df)


# ---------------------------------------------------------------------------
# Test 1 — Correct epoch size for 50:50 ratio
# ---------------------------------------------------------------------------

class TestEpochSize:
    def test_epoch_size_50_50(self) -> None:
        ds = _FakeFraudDataset(n_genuine=200, n_suspicious=50)
        sampler = BalancedEpochSampler(ds, suspicious_pct=0.5, seed=42)
        # 50:50 → n_gen_per_epoch == n_susp == 50 → total 100
        assert sampler.n_genuine_per_epoch == 50
        assert sampler.epoch_size == 100
        indices = list(sampler)
        assert len(indices) == 100

    def test_len_matches_epoch_size(self) -> None:
        ds = _FakeFraudDataset(n_genuine=500, n_suspicious=100)
        sampler = BalancedEpochSampler(ds, suspicious_pct=0.5, seed=0)
        assert len(sampler) == sampler.epoch_size


# ---------------------------------------------------------------------------
# Test 2 — All suspicious indices present in EVERY epoch (fixed, no change)
# ---------------------------------------------------------------------------

class TestSuspiciousFixed:
    def test_suspicious_indices_fixed_every_epoch(self) -> None:
        n_genuine, n_susp = 300, 60
        ds = _FakeFraudDataset(n_genuine=n_genuine, n_suspicious=n_susp)
        sampler = BalancedEpochSampler(ds, suspicious_pct=0.5, seed=42)

        # Ground-truth suspicious indices in the fake dataset
        susp_indices = set(range(n_genuine, n_genuine + n_susp))

        for run in range(3):  # Check across 3 epochs
            epoch_indices = set(sampler)
            missing = susp_indices - epoch_indices
            assert not missing, (
                f"Epoch {run}: {len(missing)} suspicious indices missing: {missing}"
            )

    def test_suspicious_indices_count_constant(self) -> None:
        ds = _FakeFraudDataset(n_genuine=400, n_suspicious=80)
        sampler = BalancedEpochSampler(ds, suspicious_pct=0.5, seed=7)
        susp_indices = set(range(400, 480))
        for _ in range(4):
            epoch_set = set(sampler)
            found_susp = epoch_set & susp_indices
            assert len(found_susp) == 80, (
                f"Expected 80 suspicious indices, got {len(found_susp)}"
            )


# ---------------------------------------------------------------------------
# Test 3 — Genuine indices CHANGE across epochs
# ---------------------------------------------------------------------------

class TestGenuineChanges:
    def test_genuine_indices_change_across_epochs(self) -> None:
        ds = _FakeFraudDataset(n_genuine=500, n_suspicious=50)
        sampler = BalancedEpochSampler(ds, suspicious_pct=0.5, seed=42)

        susp_indices = set(range(500, 550))

        genuine_epoch1 = set(sampler) - susp_indices
        genuine_epoch2 = set(sampler) - susp_indices
        genuine_epoch3 = set(sampler) - susp_indices

        # Different epochs must produce different genuine samples
        assert genuine_epoch1 != genuine_epoch2, (
            "Epoch 1 and epoch 2 genuine sets are identical — sampler is not rotating!"
        )
        assert genuine_epoch2 != genuine_epoch3, (
            "Epoch 2 and epoch 3 genuine sets are identical — sampler is not rotating!"
        )

    def test_genuine_no_duplicates_within_epoch(self) -> None:
        """Genuine images must not repeat within a single epoch (sampled without replacement)."""
        ds = _FakeFraudDataset(n_genuine=300, n_suspicious=50)
        sampler = BalancedEpochSampler(ds, suspicious_pct=0.5, seed=1)
        susp_indices = set(range(300, 350))

        for _ in range(2):
            epoch_indices = list(sampler)
            gen_indices = [i for i in epoch_indices if i not in susp_indices]
            assert len(gen_indices) == len(set(gen_indices)), (
                "Duplicate genuine indices found within a single epoch."
            )


# ---------------------------------------------------------------------------
# Test 4 — All four target ratios produce correct epoch sizes
# ---------------------------------------------------------------------------

class TestAllFourRatios:
    @pytest.mark.parametrize(
        "suspicious_pct, expected_gen, expected_total",
        [
            (0.5, 325, 650),    # 50:50  → 325 * (0.5/0.5) = 325 gen
            (0.4, 488, 813),    # 40:60  → 325 * (0.6/0.4) = 487.5 → round() = 488
            (0.3, 758, 1083),   # 30:70  → 325 * (0.7/0.3) = 758.3 → round() = 758
            (0.2, 1300, 1625),  # 20:80  → 325 * (0.8/0.2) = 1300
        ],
    )
    def test_ratio_epoch_sizes(
        self, suspicious_pct: float, expected_gen: int, expected_total: int
    ) -> None:
        """Verify epoch sizes match the plan's calculation for all four ratios."""
        # Use exactly 325 suspicious (matching real train manifest)
        ds = _FakeFraudDataset(n_genuine=5329, n_suspicious=325)
        sampler = BalancedEpochSampler(ds, suspicious_pct=suspicious_pct, seed=42)

        assert sampler.n_genuine_per_epoch == expected_gen, (
            f"suspicious_pct={suspicious_pct}: expected n_gen={expected_gen}, "
            f"got {sampler.n_genuine_per_epoch}"
        )
        assert sampler.epoch_size == expected_total, (
            f"suspicious_pct={suspicious_pct}: expected epoch_size={expected_total}, "
            f"got {sampler.epoch_size}"
        )
        # Verify __iter__ yields correct count
        assert len(list(sampler)) == expected_total

    def test_invalid_suspicious_pct_raises(self) -> None:
        ds = _FakeFraudDataset()
        with pytest.raises(ValueError, match="suspicious_pct must be in"):
            BalancedEpochSampler(ds, suspicious_pct=0.0)
        with pytest.raises(ValueError, match="suspicious_pct must be in"):
            BalancedEpochSampler(ds, suspicious_pct=1.5)

    def test_epoch_counter_increments(self) -> None:
        ds = _FakeFraudDataset(n_genuine=200, n_suspicious=50)
        sampler = BalancedEpochSampler(ds, suspicious_pct=0.5, seed=42)
        assert sampler.epoch == 0
        list(sampler)
        assert sampler.epoch == 1
        list(sampler)
        assert sampler.epoch == 2
