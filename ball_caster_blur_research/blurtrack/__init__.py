"""Isolated learned-tracking experiments with the accepted physical backend."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT.parent / 'ball_caster_dual_cam'
# Import the immutable accepted geometry; all writes are controlled here.
if str(BASELINE) not in sys.path:
    sys.path.insert(0, str(BASELINE))

