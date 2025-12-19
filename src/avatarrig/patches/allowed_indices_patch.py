from __future__ import annotations

from typing import Dict, Any, Optional, Sequence, Set, Tuple
import numpy as np

# Key landmark indices from MediaPipe PoseLandmarker (33 landmarks)
# Upper body anchors
UPPER = {0, 11, 12, 13, 14, 15, 16}  # nose + shoulders/elbows/wrists
# Lower body anchors (include feet)
LOWER = {23, 24, 25, 26, 27, 28, 29, 30, 31, 32}  # hips/knees/ankles/heels/foot_index

# Bone segments used to define "plausible support"
UPPER_BONES = [
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 12),  # shoulder width
]
LOWER_BONES = [
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (23, 24),  # hip width
    (27, 29), (27, 31),
    (28, 30), (28, 32),
]
TORSO_BONES = [
    (11, 23), (12, 24),
]

def classify_frame_region_from_mask_stats(seg_stats: Optional[Dict[str, Any]], h: int) -> str:
    """
    Classify frame into 'full', 'upper', or 'lower' using segmentation bbox.
    - If the mask bbox height is small (close-up), decide upper vs lower by bbox center.
    - Otherwise treat as full.
    """
    if not seg_stats or h <= 0:
        return "full"

    bbox = seg_stats.get("bbox_xyxy_px")
    if not bbox or len(bbox) != 4:
        return "full"

    x0, y0, x1, y1 = bbox
    bbox_h = max(0.0, float(y1) - float(y0))
    frac_h = bbox_h / float(h)
    cy = (float(y0) + float(y1)) / 2.0 / float(h)

    # Heuristic: close-up / partial if bbox height occupies less than ~45% of the image
    if frac_h < 0.45:
        return "lower" if cy > 0.55 else "upper"
    return "full"

def _valid_point(
    lm_px: Sequence[Dict[str, Any]],
    confs: Sequence[float],
    i: int,
    thr: float,
    mask_u8: Optional[np.ndarray],
    w: int,
    h: int,
) -> Optional[Tuple[float, float]]:
    if i < 0 or i >= len(lm_px) or i >= len(confs):
        return None
    c = float(confs[i])
    if c < thr:
        return None
    x = float(lm_px[i].get("x_px", -1))
    y = float(lm_px[i].get("y_px", -1))
    xi = int(round(x)); yi = int(round(y))
    if xi < 0 or yi < 0 or xi >= w or yi >= h:
        return None
    if mask_u8 is not None and mask_u8[yi, xi] == 0:
        return None
    return (x, y)

def allowed_indices_from_pose(
    lm_px: Sequence[Dict[str, Any]],
    confs: Sequence[float],
    mask_u8: Optional[np.ndarray],
    thr: float,
    *,
    w: Optional[int] = None,
    h: Optional[int] = None,
    region: str = "full",
    min_len_px: float = 18.0,
) -> Set[int]:
    """
    Structural gating for overlays + bone evidence.
    - Only accept joints that participate in plausible segments (min length, endpoints valid).
    - On partial frames, restrict accepted joints to region-appropriate sets (upper vs lower).
    - Explicitly avoid 'shoulders+hips only' hallucinations by requiring limb support.
    """
    if not lm_px or len(lm_px) < 33:
        return set()

    if h is None or w is None:
        if mask_u8 is not None:
            h, w = mask_u8.shape[:2]
        else:
            return set()

    bones = []
    allowed_pool: Set[int] = set(range(33))

    if region == "upper":
        bones = UPPER_BONES
        allowed_pool = set(UPPER) | {23, 24}  # hips only as weak anchors
    elif region == "lower":
        bones = LOWER_BONES
        allowed_pool = set(LOWER) | {11, 12}  # shoulders only as weak anchors
    else:
        bones = UPPER_BONES + LOWER_BONES + TORSO_BONES
        allowed_pool = set(range(33))

    keep: Set[int] = set()

    for a, b in bones:
        if a not in allowed_pool or b not in allowed_pool:
            continue
        pa = _valid_point(lm_px, confs, a, thr, mask_u8, w, h)
        pb = _valid_point(lm_px, confs, b, thr, mask_u8, w, h)
        if pa is None or pb is None:
            continue
        d = float(((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5)
        if d < min_len_px:
            continue
        keep.add(a); keep.add(b)

    has_upper_limb = any(i in keep for i in (13, 14, 15, 16))
    has_lower_limb = any(i in keep for i in (25, 26, 27, 28, 29, 30, 31, 32))

    if region == "upper":
        if not has_upper_limb:
            keep -= {11, 12, 23, 24}
    elif region == "lower":
        if not has_lower_limb:
            keep -= {11, 12, 23, 24}

    if not keep:
        salvage = LOWER if region == "lower" else UPPER
        for i in salvage:
            p = _valid_point(lm_px, confs, i, thr, mask_u8, w, h)
            if p is not None:
                keep.add(i)

    keep &= allowed_pool
    return keep
