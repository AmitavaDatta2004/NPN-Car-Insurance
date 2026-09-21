"""pytest configuration and path setup for ClaimVision ML tests."""

import sys
from pathlib import Path

# Ensure ml/src is on sys.path regardless of how pytest is invoked
ml_src = Path(__file__).resolve().parent.parent / "src"
if ml_src.exists() and str(ml_src) not in sys.path:
    sys.path.insert(0, str(ml_src))
