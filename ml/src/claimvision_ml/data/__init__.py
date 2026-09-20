"""data — dataset loading, manifest reading, audit, and split utilities.

Audit tools for dataset verification, exact and perceptual duplicate detection,
shortcut risk analysis, and group-aware split generation with zero leakage.
"""

from claimvision_ml.data.audit import (
    audit_image_files,
    audit_image_quality_distribution,
    compute_file_hashes,
    compute_file_sha256,
    compute_perceptual_hashes,
    detect_shortcut_risks,
    find_exact_duplicate_groups,
    load_and_validate_csv,
)
from claimvision_ml.data.manifest import (
    create_group_aware_split,
    save_manifests,
    validate_split_leakage,
)

__all__ = [
    "load_and_validate_csv",
    "audit_image_files",
    "compute_file_sha256",
    "compute_file_hashes",
    "find_exact_duplicate_groups",
    "compute_perceptual_hashes",
    "audit_image_quality_distribution",
    "detect_shortcut_risks",
    "create_group_aware_split",
    "validate_split_leakage",
    "save_manifests",
]
