"""Tagged-fork and speckled-sidewall swivel-caster tracking."""

from .adapter import SwivelSeries, adapt_swivel_series
from .geometry import (
    SidewallProjection,
    SwivelGeometry,
    project_sidewall,
    projected_sidewall_mask,
    unproject_to_plane,
)
from .pipeline import SwivelPipelineResult, run_clip, run_pipeline
from .reference import (
    ReferenceObservation,
    detect_reference_phase,
    unwrap_reference_phase,
)
from .roll import (
    ImagePlaneRollCheck,
    RollEstimate,
    RollEstimator,
    estimate_roll_increment,
    image_plane_circular_check,
)
from .tag import (
    ArucoTagTracker,
    TagDetector,
    TagObservation,
    differentiate_angles_with_gaps,
    generate_marker_image,
    relative_swivel_yaw,
    unwrap_angles_with_gaps,
)

__all__ = [
    "ArucoTagTracker",
    "ImagePlaneRollCheck",
    "ReferenceObservation",
    "RollEstimate",
    "RollEstimator",
    "SidewallProjection",
    "SwivelGeometry",
    "SwivelPipelineResult",
    "SwivelSeries",
    "TagDetector",
    "TagObservation",
    "adapt_swivel_series",
    "detect_reference_phase",
    "differentiate_angles_with_gaps",
    "estimate_roll_increment",
    "generate_marker_image",
    "image_plane_circular_check",
    "project_sidewall",
    "projected_sidewall_mask",
    "relative_swivel_yaw",
    "run_clip",
    "run_pipeline",
    "unproject_to_plane",
    "unwrap_angles_with_gaps",
    "unwrap_reference_phase",
]
