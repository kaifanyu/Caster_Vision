# Put your inputs here

This directory deliberately contains no fabricated calibration or footage.

- Put the real test video at `data/clip.mp4` (or change `input.path` in
  `config.yaml`). Videos and image-sequence directories/globs are supported.
- Enter the camera matrix and distortion coefficients directly in
  `config.yaml`, or run `scripts/calibrate_camera.py` on checkerboard images to
  write them for you.
- For axis calibration, record one smooth 60-90 degree sweep in each direction:
  `axis_calibration/pure_roll.mp4` for roll and
  `axis_calibration/pure_swivel.mp4` for swivel. Do not pause, reverse, or move
  the camera/caster housing between these clips and the main clip. Then run:

  ```powershell
  python .\scripts\calibrate_axes.py --config .\config.yaml --roll .\data\axis_calibration\pure_roll.mp4 --swivel .\data\axis_calibration\pure_swivel.mp4 --min-axis-step-deg 0.20
  ```

  Continue only after it prints `PASS`; the report includes raw, rejected
  low-motion, and retained increment counts. Then generate a non-strict
  baseline for `data/clip.mp4` with:

  ```powershell
  python .\scripts\run.py --config .\config.yaml
  ```

  Inspect `out/real/tracking_overlay.mp4` and the terminal quality summary
  before rerunning with `--strict`. See the root README for interpretation.

Do not commit private or large footage by accident; common video files and the
`data/frames/` directory are ignored by Git.
