# Camera readiness check — 2026-09-16

**Manual controls verified. The rig is not yet ready for reliable axis calibration.**
The Brio intermittently emits incomplete JPEG frames, and the C920 green shell
needs more illumination. A stream-only test passed for 30 seconds, but the final
30-second recording request failed at about 18.4 seconds, so it is not a pass.

## Saved settings

The profiles are in [config/rig.yaml](../config/rig.yaml). Use `scripts/record.py`
for both calibration and measurements; it reapplies and checks these settings.
Other camera applications can overwrite them.

| Setting | C920 | Brio 101 |
|---|---|---|
| Image format | 1920×1080 MJPG, requested 30 fps | 1920×1080 MJPG, requested 30 fps |
| Focus | Autofocus off; retained manual position 50; zoom 100 | Fixed-focus lens; no autofocus control |
| Exposure | Auto exposure off; 7700 µs (V4L2 value 77) | Auto exposure off; 8000 µs (V4L2 value 80) |
| Gain | Fixed 180 | Fixed 180 |
| White balance | Auto off; fixed 4000 K | Auto off; fixed 4000 K |
| Dynamic frame rate | Off | Off |
| Anti-flicker menu | 60 Hz | 60 Hz |
| Brightness/contrast/saturation/sharpness | 128 each | 128 each |
| Backlight compensation | 0 | 0 |
| Capture buffers | 4 | 4 |

All requested controls matched after startup, after warmup, and after recording.
The final independent [control readback](../out/camera_readiness_20260916_214213/final_controls.json) also matched.
Focus position 50 was preserved from the existing C920 setup, not claimed to be
an optical optimum. About 8 ms exposure was selected because 4 ms was too dark
under the present lighting. Faster motion needs more light and a shorter tested
exposure. C920 exposure 4000 µs was observed to settle to 3800 µs during streaming;
7700 µs held the requested value.

## Recording and integrity result

| Final saved prefix | C920 | Brio 101 |
|---|---:|---:|
| Valid frames saved | 536 | 551 |
| Observed receive rate | 29.05 fps | 29.93 fps |
| Intervals over 50 ms | 19 | 0 |

The Brio then returned a JPEG without its end-of-image marker. An independent
18-second paired packet test reproduced one such packet among 540 Brio frames.
That saved example has **no EOI anywhere**, not merely extra bytes after an EOI.
The fault's USB/driver/firmware source has not been isolated. Four buffers passed
one stream-only test but did not eliminate the failure in the saved-video test.

The recorder stops, finalizes the earlier valid AVI frames, and marks the session
`failed`; processing refuses that failed session. It does not append a fabricated
EOI or quietly accept recovered pixels. All 1087 saved valid-prefix frames were decoded
successfully at 1920×1080, and each AVI count matched its CSV rows. FFmpeg emitted
APP-metadata warnings while decoding the native Brio frames; their pixel arrays
and frame counts were available, but the metadata was not error-free. This validates
the writer's output but does not turn the failed capture into a calibration clip.

Evidence: [machine-readable report](../out/camera_readiness_20260916_214213/readiness.json),
[final session manifest](../out/camera_readiness_20260916_214213/paired_final_four_buffers/session.json),
[paired packet diagnostic](../out/camera_readiness_20260916_214213/paired_packet_test.json),
[30-second four-buffer diagnostic](../out/camera_readiness_20260916_214213/paired_four_buffer_test.json).

## Changes made during validation

- Reapply controls after the first streaming frame. C920 exposure changed from
  the requested 4 ms to 31.2 ms on initial STREAMON despite correct earlier readback.
- Drain and verify warmup on both cameras before starting the shared recording clock.
- Save original MJPEG payloads directly into indexed AVI without JPEG re-encoding.
  The former decoded/re-encoded recorder achieved only about 12–13 fps here.
- Use four capture buffers. One buffer limited the Brio to roughly half its requested
  rate; two restored approximately 30 fps, and four provide more queue headroom.
- Accept the C920's measured 0–31 bytes of native buffer padding after EOI, while
  requiring actual JPEG SOI/SOF/EOI and matching dimensions.
- Save reviewed caster circles and color thresholds. Automatic circles selected
  background regions in both views. These image regions depend on the current mounts.

The AVI muxer has a roughly 4 GiB per-camera, per-session limit. Files retain
original JPEG data without image resizing. CSV receive timestamps remain the
time source; nominal AVI playback rate is not evidence of synchronized exposure.

## Image quality and remaining work

The reviewed static image had 80 red / 29 green corners in the C920 and 80 / 80
in the Brio. About 22 of the C920's green corners cluster on one small bright
region. More diffuse light on that green shell is needed before relying on its
axis constraints. Static feature counts do not measure motion accuracy or blur.

Images: [C920 frame](../out/camera_readiness_20260916_214213/c920_final_frame.png), [Brio frame](../out/camera_readiness_20260916_214213/brio101_final_frame.png),
[C920 feature overlay](../out/camera_readiness_20260916_214213/static_quality_c920_reviewed.png),
[Brio feature overlay](../out/camera_readiness_20260916_214213/static_quality_brio101_reviewed.png),
[detailed static feature check](../out/camera_readiness_20260916_214213/static_quality_report.md).

1. Improve diffuse illumination of the C920 green shell, then recheck the masks
   with the saved manual controls.
2. Investigate the Brio stream: reconnect directly to another USB port, check
   kernel USB/UVC errors, and repeat a full paired recording. Do not treat a
   successful short preview as proof that the intermittent failure is gone.
3. Verify the supplied intrinsics against a checkerboard at these settings.
   Both YAMLs retain `capture_profile: null` because their original capture
   settings are unknown; this hardware check does not establish geometric accuracy.
4. Obtain stereo extrinsics from simultaneous checkerboard views, measure relative
   exposure timing with a shared event, then record pure-roll and pure-swivel sessions.
   Stereo extrinsics and timing verification are still unset.

Repeat a capture check from the repository root, choosing a new output directory:

```bash
python3 scripts/record.py --mode motion --duration 30 --output recordings/readiness_recheck_01
```

For calibration video commands, see [CAMERA_SETUP.md](CAMERA_SETUP.md).

Software regression check: `python3 -m pytest -q` — **73 passed in 21.36 s**.
