#!/usr/bin/env python
"""CLI wrapper for all S-A through S-E swivel gates."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synthetic.validate_synthetic_swivel import main


if __name__ == "__main__":
    raise SystemExit(main())
