
### 1. Convert all three original videos to upright MKVs
Rotates by 180 degrees and creates it into data/upright/clip.mkv

```powershell
Set-Location 'C:\Users\kyu\Documents\Upenn\Caster_Vision\ball_caster_rot'
New-Item -ItemType Directory -Force -Path .\data\upright | Out-Null

if (-not (Test-Path -LiteralPath .\data\upright\clip.mkv)) {
    ffmpeg -hide_banner -nostdin -n -noautorotate -i .\data\clip.mp4 -map 0:v:0 -vf "format=bgr0,hflip,vflip" -fps_mode passthrough -enc_time_base:v demux -c:v ffv1 -level 3 -coder 1 -context 1 -g 1 -pix_fmt bgr0 -slicecrc 1 -an .\data\upright\clip.mkv
    if ($LASTEXITCODE -ne 0) { throw 'Main-clip conversion failed. Inspect the output before retrying.' }
}

if (-not (Test-Path -LiteralPath .\data\upright\pure_swivel.mkv)) {
    ffmpeg -hide_banner -nostdin -n -noautorotate -i .\data\axis_calibration\pure_swivel.mp4 -map 0:v:0 -vf "format=bgr0,hflip,vflip" -fps_mode passthrough -enc_time_base:v demux -c:v ffv1 -level 3 -coder 1 -context 1 -g 1 -pix_fmt bgr0 -slicecrc 1 -an .\data\upright\pure_swivel.mkv
    if ($LASTEXITCODE -ne 0) { throw 'Pure-swivel conversion failed. Inspect the output before retrying.' }
}

if (-not (Test-Path -LiteralPath .\data\upright\pure_roll.mkv)) {
    ffmpeg -hide_banner -nostdin -n -noautorotate -i .\data\axis_calibration\pure_roll.mp4 -map 0:v:0 -vf "format=bgr0,hflip,vflip" -fps_mode passthrough -enc_time_base:v demux -c:v ffv1 -level 3 -coder 1 -context 1 -g 1 -pix_fmt bgr0 -slicecrc 1 -an .\data\upright\pure_roll.mkv
    if ($LASTEXITCODE -ne 0) { throw 'Pure-roll conversion failed. Inspect the output before retrying.' }
}
```
Frame Check

```powershell
foreach ($clip in @('clip', 'pure_swivel', 'pure_roll')) {
    ffprobe -v error -select_streams v:0 -count_frames -show_entries format=filename:stream=codec_name,width,height,nb_read_frames -of json ".\data\upright\$clip.mkv"
    if ($LASTEXITCODE -ne 0) { throw "Cannot inspect upright $clip.mkv" }
}
```

### 2. Confirm the camera calibration and input paths

`config.yaml` now contains:

```yaml
input:
  path: data/upright/clip.mkv
  type: auto
  max_frames: null
  fps_override: null
```

### 3. Fit Circle on starting image
Click 3 separated points along the edge.

```powershell
.\.venv\Scripts\python.exe .\scripts\calibrate_circle.py --config .\config.yaml --manual --preview .\out\upright_annotation\circle_preview.png
```


### 4. Annotate colors and save HSV thresholds

```powershell
.\.venv\Scripts\python.exe .\scripts\inspect_hsv.py --config .\config.yaml --frames 12 --write
```

- `T`: sample the red physical shell (`top`).
- `B`: sample the green physical shell (`bottom`).
- `Y`: sample the dark yoke.
- `N`/`P` Next frame / Previous frame
- `U` undo
- `R` resets that class.

### 5. Calibrate axes using both upright motion clips


```powershell
.\.venv\Scripts\python.exe .\scripts\calibrate_axes.py --config .\config.yaml --roll .\data\upright\pure_roll.mkv --swivel .\data\upright\pure_swivel.mkv --min-axis-step-deg 0.20 --max-frames 120 --report .\out\upright_annotation\axis_calibration_first120_report.json
```


### 6. Run tracking after calibration passes

```powershell
.\.venv\Scripts\python.exe .\scripts\run.py --config .\config.yaml
```


### 7. Apply Kalman filter and render axis

```
.\scripts\fuse_motion.py --results .\out\tracked\results.json --output .\out\fused
```
