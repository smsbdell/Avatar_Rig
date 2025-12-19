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
