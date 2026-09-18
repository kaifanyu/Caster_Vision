"""Planning approximation, not a calibrated blur measurement of the caster."""
from __future__ import annotations

import csv
import math
from pathlib import Path


def main():
    destination = Path(__file__).resolve().parents[1] / 'reports'
    destination.mkdir(exist_ok=True)
    rows = []
    radius_px = 500.  # representative image lever arm, not curvature radius in meters
    for degrees_s in (90., 180., 360., 720., 1080.):
        speed = radius_px * math.radians(degrees_s)
        for exposure_ms in (15.6, 8., 2., 1., .5):
            rows.append({'speed_deg_s': degrees_s, 'image_lever_arm_px': radius_px,
                         'exposure_ms': exposure_ms, 'angular_sweep_deg': degrees_s*exposure_ms/1000.,
                         'approx_blur_px': speed*exposure_ms/1000.,
                         'exposure_for_2px_ms': 2000./speed,
                         'step_deg_at_30fps': degrees_s/30.,
                         'step_deg_at_120fps': degrees_s/120.})
    output = destination/'exposure_budget.csv'
    with output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    print(f'Wrote {output}; use calibrated projection derivatives for an actual surface point.')


if __name__ == '__main__':
    main()
