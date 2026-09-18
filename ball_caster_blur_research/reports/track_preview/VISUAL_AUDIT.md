# CoTracker image audit

Inspected raw/overlay pairs at 1.0, 11.2, 19.2, and 21.2 seconds after both camera collectors finished. Images use original native frames and original calibrated undistortion. The collector window nearest each selected frame's temporal center supplies the dots.

| Requested time | C920 red / green candidates | Brio101 red / green candidates |
|---:|---:|---:|
| 1.0 s | 91 / 89 | 94 / 92 |
| 11.2 s | 24 / 20 | 52 / 3 |
| 19.2 s | 1 / 75 | 0 / 68 |
| 21.2 s | 0 / 3 | 3 / 56 |

These are window candidates before spatial deduplication and physical residual/visibility checks, so they are not counts of independent accepted physical measurements. At 1.0 s, 64 Brio candidates coincide with detected query anchors; white rings identify those observations. Other displayed frames have no query anchors.

- Early accepted dots generally align with painted triangle corners and shell texture. The displayed frames do not show broad locking onto the black yoke.
- At 11.2 s, C920 retains candidates on both shells; Brio mainly sees the red shell. The green shell is narrow in the Brio view and blurred in C920.
- At 19.2 s, the red shell is heavily blurred in both cameras. Only one C920 red candidate and no Brio red candidates survive the collector. Both cameras retain numerous green candidates.
- At 21.2 s, Brio supplies many green candidates while C920 retains only three. The three Brio red candidates lie on blurred edge/streak locations; this image alone cannot establish their material identity or red angular velocity.

The two cameras can complement visibility, but two blurred views do not automatically recover the missing red spin. Still-image agreement does not establish temporal material identity. Downstream held-out pixel consistency, physical visibility, and measurement-only rate support remain necessary.

Artifacts: [contact sheet](track_contacts.jpg), [1.0 s raw/overlay](tracks_00_01.000s.jpg), [11.2 s](tracks_01_11.200s.jpg), [19.2 s](tracks_02_19.200s.jpg), [21.2 s](tracks_03_21.200s.jpg), [exact frames, times, windows, and counts](track_preview.json).
