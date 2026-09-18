# Reused tracking primitives

The `ballrot/` modules were copied from the neighboring `ball_caster_rot/ballrot` on 2026-09-16. They provide camera geometry, segmentation, KLT, and rotation initialization. The new joint dual-camera estimator lives in `dualcam/`; the original project is unchanged.

On 2026-09-18, `temporal.py` and `motion_recovery.py` were also copied for the
native-time shared motion tracker. `appearance.py` contains the standalone
`_patch_similarity` helper from the original `mechanical_rematch.py`; the sole
import change in `motion_recovery.py` points to this helper. The original temporal
and motion-recovery tests are retained as `test_temporal_port.py` and
`test_temporal_motion_port.py`. Test-helper imports are adjusted; image-level
pipeline tests use `tests/temporal_adapter.py` to exercise the production
`IncrementProposal` integration instead of the absent old single-camera pipeline.
No runtime dependency on the neighboring project is required.
