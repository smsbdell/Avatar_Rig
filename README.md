# AvatarRig — Phase 1 (Tasks-based)

This repo is an early phase-1 prototype that:
1) Runs high-volume landmark extraction on a dataset of real-world photos:
   - Pose landmarks (33) + optional pose segmentation mask
   - Face landmarks (478) + blendshapes (optional)
   - Person segmentation (background/person) via Image Segmenter
2) Aggregates those observations into a first-pass skeletal proportion estimate.

## Folder conventions

- `inputs/datasets/<DATASET_NAME>/...`  
  Your raw image folders (can be nested arbitrarily)

- `inputs/models/mediapipe/`  
  Model files downloaded automatically on first run (or you can place them manually)

- `outputs/runs/<RUN_NAME>/`  
  Output artifacts for each run:
  - `observations/*.json`
  - `overlays/*.png` (optional)
  - `masks/*.png` (optional)
  - `phase1_summary.json`
  - `skeleton_estimate.json`

## Quickstart (PowerShell)

```powershell
cd C:\Users\smsbd\OneDrive\Desktop\avatar_rig
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
avatarrig ".\inputs\datasets\TestRun1" --out ".\outputs\runs\TestRun1" --workers 4 --debug-overlays --save-masks
```

`pip install -e .` installs the project in *editable* mode, so running `avatarrig` uses your local source files without needing a reinstall after each edit.

### Pose profiles

Use `--pose-profile strict` to raise MediaPipe's pose detection/presence/tracking thresholds. This profile favors high-precision detections when backgrounds are cluttered or when the dataset is dominated by partial-body crops, at the cost of dropping marginal poses. The default (balanced) thresholds remain accessible through manual `--min-pose-*` values or by setting `--pose-profile balanced`.

### Calibrate thresholds from an audit set

After labeling a small audit set and running the pipeline once (with overlays/masks if helpful), aggregate the observation metrics to recommend dataset-specific thresholds. This can cut hallucinations before scaling to the full dataset.

```
python scripts/calibrate_thresholds.py outputs/runs/TestAudit --report outputs/runs/TestAudit/calibration.txt
```

The report summarizes `pose_score`, `pose_inmask_ratio`, `pose_supported_bones`, and coarse `frame_kind` counts for each run under `outputs/runs/`. It also prints suggested CLI overrides such as `--min-pose-det`, `--min-pose-presence`, `--min-pose-track`, and `--overlay-thr` that you can feed back into `avatarrig` for the next pass.

## PowerShell quick debug

To sanity-check a single image in PowerShell (with venv active):

```powershell
$code = @'
import numpy as np, cv2
from avatarrig.detectors.mediapipe_tasks_backend import MediaPipeTasksBackend
img = cv2.imread(r".\\inputs\\datasets\\Smoke\\images\\sash (1).JPG")
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
b = MediaPipeTasksBackend()
pose = b.detect_pose(rgb)
seg  = b.segment_person(rgb)
m = seg.category_mask.numpy_view()
print('pose_landmarks:', len(pose.pose_landmarks[0]) if pose.pose_landmarks else 0)
print('seg unique:', np.unique(m)[:20])
'@
$code | python
```
