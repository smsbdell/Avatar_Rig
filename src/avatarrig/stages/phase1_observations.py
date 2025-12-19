from __future__ import annotations

from pathlib import Path
import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable

import cv2
import numpy as np
from tqdm import tqdm

from ..detectors.mediapipe_tasks_backend import MediaPipeTasksBackend, MediaPipeTasksConfig, POSE_LANDMARK_NAMES
from ..schemas.observation import Observation, Status, PoseObservation, FaceObservation, SegmentationObservation, Landmark2D

IMAGE_EXTS = {".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff"}

def _iter_images(dataset_dir: Path) -> list[Path]:
    out: list[Path] = []
    for p in dataset_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out.append(p)
    out.sort()
    return out

def _safe_relpath(p: Path, root: Path) -> str:
    try:
        return p.relative_to(root).as_posix()
    except Exception:
        return p.as_posix()

def _image_id_from_rel(rel: str) -> str:
    # Stable-ish ID for filenames with weird chars.
    base = rel.replace("/", "__")
    base = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in base)
    return base.replace(".", "_")


def _choose_foreground_from_category_mask(mask: np.ndarray) -> tuple[np.ndarray, str, dict]:
    """Given a CATEGORY_MASK, choose which category corresponds to the person.

    The ImageSegmenter CATEGORY_MASK encodes the winning category index per pixel
    in the range [0,255]. For some binary person/background models, the 'person'
    label is not guaranteed to be the non-zero index in all distributions.

    Heuristic: compute both candidates (mask==0) and (mask!=0). In uncontrolled,
    real-world photos, the person typically occupies less area than the
    background. We therefore select the *smaller* region when both are plausible,
    and fall back to plausibility bounds when they are not.
    """
    if mask is None:
        raise ValueError("mask is None")

    mask = np.asarray(mask)
    if mask.ndim == 3:
        # Some builds return HxWx1 for category masks; squeeze to HxW.
        if mask.shape[2] == 1:
            mask = mask[:, :, 0]
        else:
            # If something unexpected, reduce to first channel.
            mask = mask[:, :, 0]
    elif mask.ndim != 2:
        mask = np.squeeze(mask)
        if mask.ndim != 2:
            raise ValueError(f"Unexpected CATEGORY_MASK shape: {mask.shape}")

    m0 = (mask == 0)
    m1 = (mask != 0)

    r0 = float(m0.mean())
    r1 = float(m1.mean())

    # Plausible person coverage range (loose). We keep it wide because we are
    # collecting evidence, not rejecting aggressively.
    MIN_R, MAX_R = 0.01, 0.98

    # Prefer the smaller candidate unless it is implausibly tiny or the larger
    # one is implausibly huge.
    if MIN_R <= min(r0, r1) <= MAX_R:
        if max(r0, r1) > MAX_R:
            fg, mode, ratio = (m0, "eq0", r0) if r0 < r1 else (m1, "nonzero", r1)
        else:
            fg, mode, ratio = (m0, "eq0", r0) if r0 <= r1 else (m1, "nonzero", r1)
    else:
        # If the smaller is effectively empty, use the other.
        fg, mode, ratio = (m0, "eq0", r0) if r0 > r1 else (m1, "nonzero", r1)

    stats = {"area_ratio": ratio, "cand_eq0": r0, "cand_nonzero": r1}
    return fg, mode, stats


def _allowed_pose_indices_from_bones(
    pose_lms: list[Landmark2D],
    *,
    person_mask_u8: np.ndarray | None,
    thr: float,
    w: int,
    h: int,
    min_len_px: float = 12.0,
) -> set[int]:
    """Return a set of landmark indices that are supported by credible evidence.

    Goal: suppress "hallucinated full skeleton" on partial-body frames (closeups).
    We keep only joints that:
      - are in-bounds,
      - have conservative conf >= thr, and
      - lie on the person mask (if provided),
      - and participate in at least one plausible bone segment.

    If no plausible long bones exist, we allow foot/ankle anchors as a salvage path.
    """
    if not pose_lms:
        return set()

    def lm_conf(lm: Landmark2D) -> float:
        v = float(lm.visibility or 0.0)
        p = float(lm.presence or 0.0)
        if p <= 0.0:
            p = v
        if v <= 0.0:
            v = p
        return float(min(v, p))

    def in_bounds(x: float, y: float) -> bool:
        return (0.0 <= x < float(w)) and (0.0 <= y < float(h))

    def on_mask(x: float, y: float) -> bool:
        if person_mask_u8 is None:
            return True
        xi = int(round(x))
        yi = int(round(y))
        if xi < 0 or yi < 0 or xi >= w or yi >= h:
            return False
        return bool(person_mask_u8[yi, xi] > 127)

    def pt(i: int):
        if i < 0 or i >= len(pose_lms):
            return None
        lm = pose_lms[i]
        if lm.x_px is None or lm.y_px is None:
            return None
        c = lm_conf(lm)
        if c < thr:
            return None
        x = float(lm.x_px); y = float(lm.y_px)
        if not in_bounds(x, y):
            return None
        if not on_mask(x, y):
            return None
        return x, y, c

    # Bone segments used for gating (index-based; BlazePose 33 landmarks).
    BONES = [
        (11, 12), (23, 24),        # shoulder/hip width
        (11, 13), (13, 15),        # left arm
        (12, 14), (14, 16),        # right arm
        (11, 23), (12, 24),        # torso sides
        (23, 25), (25, 27),        # left leg
        (24, 26), (26, 28),        # right leg
        (27, 29), (27, 31),        # left heel/foot-index
        (28, 30), (28, 32),        # right heel/foot-index
    ]

    keep: set[int] = set()
    for a, b in BONES:
        pa = pt(a)
        pb = pt(b)
        if pa is None or pb is None:
            continue
        d = float(((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5)
        if d < float(min_len_px):
            continue
        keep.add(a); keep.add(b)

    # If no plausible bones, salvage: allow foot/ankle points only (useful for boot closeups).
    if not keep:
        for i in (27, 28, 29, 30, 31, 32):
            if pt(i) is not None:
                keep.add(i)

    return keep


def _draw_pose_overlay(
    bgr: np.ndarray,
    pose_lms: list[Landmark2D],
    *,
    thr: float = 0.35,
    crop_bbox: tuple[int, int, int, int] | None = None,
    label_text: str | None = None,
    person_mask_u8: np.ndarray | None = None,
    allowed_idxs: set[int] | None = None,
) -> np.ndarray:
    """Debug overlay for Pose landmarks with partial-frame gating."""
    out = bgr.copy()
    H, W = out.shape[:2]

    # Pose landmark indices per MediaPipe Pose Landmarker (33 total).
    EDGES = [
        (11, 12),  # shoulders
        (23, 24),  # hips
        (11, 13), (13, 15),  # left arm
        (12, 14), (14, 16),  # right arm
        (11, 23), (12, 24),  # torso
        (23, 25), (25, 27), (27, 31),  # left leg
        (24, 26), (26, 28), (28, 32),  # right leg
        (27, 29), (28, 30),            # heels
    ]
    LABEL_IDXS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    def lm_conf(lm: Landmark2D) -> float:
        v = float(lm.visibility or 0.0)
        p = float(lm.presence or 0.0)
        if p <= 0.0:
            p = v
        if v <= 0.0:
            v = p
        return float(min(v, p))

    def on_mask(x: int, y: int) -> bool:
        if person_mask_u8 is None:
            return True
        if x < 0 or y < 0 or x >= W or y >= H:
            return False
        return bool(person_mask_u8[y, x] > 127)

    if allowed_idxs is None and person_mask_u8 is not None:
        allowed_idxs = _allowed_pose_indices_from_bones(
            pose_lms, person_mask_u8=person_mask_u8, thr=max(thr, 0.45), w=W, h=H, min_len_px=max(12.0, 0.02 * max(W, H))
        )

    pts: list[tuple[int, int] | None] = []
    for i, lm in enumerate(pose_lms):
        if allowed_idxs is not None and i not in allowed_idxs:
            pts.append(None)
            continue
        if lm.x_px is None or lm.y_px is None:
            pts.append(None)
            continue
        if lm_conf(lm) < thr:
            pts.append(None)
            continue
        x = int(round(float(lm.x_px)))
        y = int(round(float(lm.y_px)))
        if x < 0 or x >= W or y < 0 or y >= H:
            pts.append(None)
            continue
        if not on_mask(x, y):
            pts.append(None)
            continue
        pts.append((x, y))

    for a, b in EDGES:
        if a < len(pts) and b < len(pts) and pts[a] and pts[b]:
            cv2.line(out, pts[a], pts[b], (0, 255, 0), 2)

    for p in pts:
        if p:
            cv2.circle(out, p, 3, (0, 255, 0), -1)

    for i in LABEL_IDXS:
        if i < len(pts) and pts[i]:
            x, y = pts[i]
            name = POSE_LANDMARK_NAMES[i] if i < len(POSE_LANDMARK_NAMES) else str(i)
            name = name.replace("left_", "L_").replace("right_", "R_")
            label = f"{i}:{name}"
            cv2.putText(out, label, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, label, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    if crop_bbox is not None:
        x0, y0, x1, y1 = crop_bbox
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 0, 0), 2)

    if label_text:
        cv2.putText(out, label_text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, label_text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def _bbox_from_mask(fg: np.ndarray, *, pad_frac: float, w: int, h: int) -> tuple[int,int,int,int] | None:
    # Normalize mask to 2D boolean for bbox extraction (handles HxWx1 or HxWx3).
    if fg is None:
        return None
    fg = np.asarray(fg)
    if fg.ndim == 3:
        # Common case: HxWx1 or HxWx3. Reduce to 2D.
        if fg.shape[2] == 1:
            fg = fg[:, :, 0]
        else:
            fg = np.any(fg, axis=2)
    elif fg.ndim != 2:
        fg = np.squeeze(fg)
        if fg.ndim != 2:
            # Last resort: flatten extra dims by taking the first slice
            fg = fg.reshape(fg.shape[0], -1)
    ys, xs = np.where(fg)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    padx = int(bw * pad_frac)
    pady = int(bh * pad_frac)
    x0 = max(0, x0 - padx)
    y0 = max(0, y0 - pady)
    x1 = min(w - 1, x1 + padx)
    y1 = min(h - 1, y1 + pady)

    # Convert inclusive max indices to exclusive slice end coordinates.
    x1_excl = min(w, x1 + 1)
    y1_excl = min(h, y1 + 1)
    if x1_excl <= x0 or y1_excl <= y0:
        return None
    return x0, y0, x1_excl, y1_excl

# ---- pose scoring / selection helpers ----
_POSE_KEY_IDXS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

def _lm_conf_raw(lm) -> float:
    """Conservative per-landmark confidence.

    MediaPipe PoseLandmarker can return very high `presence` on partial-body frames.
    For skeleton/proportion extraction we prefer to avoid hallucinated joints, so we
    use the *minimum* of (presence, visibility), with fallbacks when one is absent.
    """
    v = float(getattr(lm, "visibility", 0.0) or 0.0)
    p = getattr(lm, "presence", None)
    p = float(p or 0.0)
    if p <= 0.0:
        p = v
    if v <= 0.0:
        v = p
    return float(min(v, p))



def _compute_bone_metrics(pose_lms: list[Landmark2D], *, w: int, h: int, names: list[str], person_bbox: tuple[int,int,int,int] | None = None, person_mask_u8: np.ndarray | None = None) -> tuple[dict[str,float], dict[str,float], dict[str,float], str | None]:
    """Compute per-image limb-length evidence.

    We are not solving animation pose here. We compute simple 2D bone lengths in
    pixels plus ratios to a reference length. These are later aggregated across
    many images to estimate canonical proportions.
    """
    name_to_idx = {n: i for i, n in enumerate(names or [])}

    # Person bbox used for plausibility heuristics (prefer segmentation-derived crop bbox).
    if person_bbox is not None:
        bx0, by0, bx1, by1 = person_bbox
        pbw = float(max(1, bx1 - bx0))
        pbh = float(max(1, by1 - by0))
    else:
        pbw = float(max(1, w))
        pbh = float(max(1, h))

    def _in_bounds(p) -> bool:
        if p is None:
            return False
        x, y, _ = p
        return (0.0 <= x < float(w)) and (0.0 <= y < float(h))

    def pt(i: int):
        if i < 0 or i >= len(pose_lms):
            return None
        lm = pose_lms[i]
        x = lm.x_px if lm.x_px is not None else (lm.x * w)
        y = lm.y_px if lm.y_px is not None else (lm.y * h)
        return float(x), float(y), lm

    def conf(lm: Landmark2D) -> float:
        v = float(lm.visibility or 0.0)
        p = float(lm.presence or 0.0)
        if p <= 0.0:
            p = v
        if v <= 0.0:
            v = p
        return float(min(v, p))

    def dist(a, b) -> float | None:
        if a is None or b is None:
            return None
        (ax, ay, lm_a) = a
        (bx, by, lm_b) = b
        return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)

    def bone(a_name: str, b_name: str):
        ai = name_to_idx.get(a_name)
        bi = name_to_idx.get(b_name)
        if ai is None or bi is None:
            return None, None
        pa = pt(ai)
        pb = pt(bi)
        if not _in_bounds(pa) or not _in_bounds(pb):
            return None, None
        if person_mask_u8 is not None:
            ax, ay, _ = pa
            bx, by, _ = pb
            axi = int(round(ax)); ayi = int(round(ay))
            bxi = int(round(bx)); byi = int(round(by))
            if (axi < 0 or ayi < 0 or axi >= w or ayi >= h or person_mask_u8[ayi, axi] <= 127):
                return None, None
            if (bxi < 0 or byi < 0 or bxi >= w or byi >= h or person_mask_u8[byi, bxi] <= 127):
                return None, None
        d = dist(pa, pb)
        if d is None:
            return None, None
        # Reject collapsed segments (common hallucination in partial-body frames).
        min_len = float(max(8.0, 0.02 * max(pbw, pbh)))
        if float(d) < min_len:
            return None, None
        c = float(min(conf(pa[2]), conf(pb[2])))
        if c < 0.25:
            return None, None
        return d, c

    bones_def = {
        "upper_arm_L": ("left_shoulder", "left_elbow"),
        "forearm_L": ("left_elbow", "left_wrist"),
        "upper_arm_R": ("right_shoulder", "right_elbow"),
        "forearm_R": ("right_elbow", "right_wrist"),
        "thigh_L": ("left_hip", "left_knee"),
        "shin_L": ("left_knee", "left_ankle"),
        "thigh_R": ("right_hip", "right_knee"),
        "shin_R": ("right_knee", "right_ankle"),
        "shoulder_width": ("left_shoulder", "right_shoulder"),
        "hip_width": ("left_hip", "right_hip"),
        "wingspan": ("left_wrist", "right_wrist"),
    }

    bones_px: dict[str, float] = {}
    bones_conf: dict[str, float] = {}

    for k, (a, b) in bones_def.items():
        d, c = bone(a, b)
        if d is not None:
            bones_px[k] = float(d)
            bones_conf[k] = float(c)

    # Torso (mid-shoulders -> mid-hips)
    ls = name_to_idx.get("left_shoulder"); rs = name_to_idx.get("right_shoulder")
    lh = name_to_idx.get("left_hip"); rh = name_to_idx.get("right_hip")
    if None not in (ls, rs, lh, rh):
        pls = pt(ls); prs = pt(rs); plh = pt(lh); prh = pt(rh)
        if all(p is not None for p in (pls, prs, plh, prh)):
            shx = (pls[0] + prs[0]) * 0.5
            shy = (pls[1] + prs[1]) * 0.5
            hpx = (plh[0] + prh[0]) * 0.5
            hpy = (plh[1] + prh[1]) * 0.5
            torso = float(((shx - hpx) ** 2 + (shy - hpy) ** 2) ** 0.5)
            c = float(min(conf(pls[2]), conf(prs[2]), conf(plh[2]), conf(prh[2])))
            bones_px["torso"] = torso
            bones_conf["torso"] = c

    # Choose a reference bone for ratios.
    # IMPORTANT: MediaPipe can be over-confident on left/right joints in
    # partial-body / heavily occluded images. We therefore apply lightweight
    # geometric plausibility heuristics so a collapsed shoulder width (etc.)
    # does not become the normalization anchor and explode all ratios.
    def _plausible_ref(name: str, length_px: float) -> bool:
        if length_px <= 1e-6:
            return False
        if name in ("shoulder_width", "hip_width"):
            return (0.08 * pbw) <= length_px <= (1.20 * pbw)
        if name == "wingspan":
            return (0.15 * pbw) <= length_px <= (2.20 * pbw)
        if name == "torso":
            return (0.15 * pbh) <= length_px <= (1.50 * pbh)
        return True

    ref_candidates = ["shoulder_width", "torso", "hip_width", "wingspan"]
    ref = None
    for r in ref_candidates:
        if r in bones_px and bones_conf.get(r, 0.0) >= 0.20 and _plausible_ref(r, float(bones_px[r])):
            ref = r
            break

    bones_ratio: dict[str, float] = {}
    if ref is not None:
        ref_len = bones_px[ref]
        for k, v in bones_px.items():
            bones_ratio[k] = float(v / ref_len) if ref_len > 1e-6 else 0.0

    return bones_px, bones_ratio, bones_conf, ref

def _score_pose_landmarks(
    lms,
    *,
    crop_x0: int,
    crop_y0: int,
    crop_w: int,
    crop_h: int,
    full_w: int,
    full_h: int,
    person_mask_u8: np.ndarray | None,
) -> dict:
    confs = []
    inside = 0
    used = 0
    oob = 0
    for i in _POSE_KEY_IDXS:
        if i >= len(lms):
            continue
        lm = lms[i]
        c = _lm_conf_raw(lm)
        confs.append(c)

        x_px = float(lm.x * crop_w + crop_x0)
        y_px = float(lm.y * crop_h + crop_y0)

        if x_px < 0 or x_px >= full_w or y_px < 0 or y_px >= full_h:
            oob += 1
            continue

        if c >= 0.15:
            used += 1
            if person_mask_u8 is not None:
                xi = int(round(x_px))
                yi = int(round(y_px))
                if person_mask_u8[yi, xi] > 127:
                    inside += 1

    mean_conf = float(sum(confs) / max(1, len(confs)))
    oob_ratio = float(oob / max(1, len(_POSE_KEY_IDXS)))

    # Geometric sanity: MediaPipe can be over-confident in cases of occlusion,
    # extreme crop, or when only part of the body is visible. Penalize poses
    # that imply a collapsed shoulder/hip width relative to the crop.
    def _pt_full(idx: int):
        if idx >= len(lms):
            return None
        lm = lms[idx]
        x_px = float(lm.x * crop_w + crop_x0)
        y_px = float(lm.y * crop_h + crop_y0)
        if x_px < 0 or x_px >= full_w or y_px < 0 or y_px >= full_h:
            return None
        c = _lm_conf_raw(lm)
        return x_px, y_px, c

    geom_penalty = 1.0
    geom = {}
    sh = _pt_full(11); sr = _pt_full(12)
    hh = _pt_full(23); hr = _pt_full(24)
    if sh and sr:
        shoulder_w = float(((sh[0]-sr[0])**2 + (sh[1]-sr[1])**2) ** 0.5)
        geom["shoulder_w_px"] = shoulder_w
        if shoulder_w < 0.04 * float(crop_w):
            geom_penalty *= 0.25
            geom["shoulders_collapsed"] = True
    if hh and hr:
        hip_w = float(((hh[0]-hr[0])**2 + (hh[1]-hr[1])**2) ** 0.5)
        geom["hip_w_px"] = hip_w
        if hip_w < 0.03 * float(crop_w):
            geom_penalty *= 0.5
            geom["hips_collapsed"] = True
    if sh and sr and hh and hr:
        shx = (sh[0] + sr[0]) * 0.5
        shy = (sh[1] + sr[1]) * 0.5
        hpx = (hh[0] + hr[0]) * 0.5
        hpy = (hh[1] + hr[1]) * 0.5
        torso = float(((shx-hpx)**2 + (shy-hpy)**2) ** 0.5)
        geom["torso_px"] = torso
        # If shoulders and hips end up on opposite sides of the crop, torso becomes a diagonal
        # spanning the frame; this is a frequent failure case.
        if torso > 1.15 * float(crop_h):
            geom_penalty *= 0.5
            geom["torso_diagonal"] = True

    inmask_ratio = None
    if person_mask_u8 is not None and used > 0:
        inmask_ratio = float(inside / used)

    # Bone-support signal: count plausible long bones supported by mask+confidence.
    supported_bones = 0
    support_geom = {}
    bones_for_support = [
        (11, 12, "shoulders"),
        (23, 24, "hips"),
        (11, 13, "L_upper_arm"), (13, 15, "L_forearm"),
        (12, 14, "R_upper_arm"), (14, 16, "R_forearm"),
        (23, 25, "L_thigh"), (25, 27, "L_shin"),
        (24, 26, "R_thigh"), (26, 28, "R_shin"),
    ]
    min_len = float(max(8.0, 0.02 * max(crop_w, crop_h)))

    def _on_mask(x: float, y: float) -> bool:
        if person_mask_u8 is None:
            return True
        xi = int(round(x)); yi = int(round(y))
        if xi < 0 or yi < 0 or xi >= full_w or yi >= full_h:
            return False
        return bool(person_mask_u8[yi, xi] > 127)

    for a, b, name in bones_for_support:
        pa = _pt_full(a)
        pb = _pt_full(b)
        if not (pa and pb):
            continue
        if min(pa[2], pb[2]) < 0.35:
            continue
        if (not _on_mask(pa[0], pa[1])) or (not _on_mask(pb[0], pb[1])):
            continue
        d = float(((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2) ** 0.5)
        if d < min_len:
            continue
        supported_bones += 1
    support_score = float(min(1.0, supported_bones / 4.0))

    score = mean_conf
    if inmask_ratio is not None:
        score = 0.35 * mean_conf + 0.45 * float(inmask_ratio) + 0.20 * support_score
    else:
        score = 0.65 * mean_conf + 0.35 * support_score

    score *= float(geom_penalty)
    score *= (1.0 - 0.75 * oob_ratio)
    score = float(max(0.0, min(1.0, score)))

    return {
        "score": score,
        "mean_conf": mean_conf,
        "inmask_ratio": inmask_ratio,
        "oob_ratio": oob_ratio,
        "geom_penalty": float(geom_penalty),
        "geom": geom,
        "used_inmask_pts": int(used),
        "supported_bones": int(supported_bones),
        "support_score": float(support_score),
    }

def _extract_best_pose_from_result(
    pose_res,
    *,
    crop_x0: int,
    crop_y0: int,
    crop_w: int,
    crop_h: int,
    full_w: int,
    full_h: int,
    person_mask_u8: np.ndarray | None,
):
    poses = getattr(pose_res, "pose_landmarks", None) or []
    if not poses:
        return None

    best = None
    best_idx = None
    best_metrics = None
    for j, lms in enumerate(poses):
        metrics = _score_pose_landmarks(
            lms,
            crop_x0=crop_x0, crop_y0=crop_y0, crop_w=crop_w, crop_h=crop_h,
            full_w=full_w, full_h=full_h,
            person_mask_u8=person_mask_u8,
        )
        if best is None or metrics["score"] > best:
            best = metrics["score"]
            best_idx = j
            best_metrics = metrics

    lms = poses[int(best_idx)]
    pose_lms: list[Landmark2D] = []
    for lm in lms:
        x_px = float(lm.x * crop_w + crop_x0)
        y_px = float(lm.y * crop_h + crop_y0)
        pose_lms.append(Landmark2D(
            x=float(x_px / full_w), y=float(y_px / full_h), z=float(getattr(lm, "z", 0.0)),
            visibility=float(getattr(lm, "visibility", 0.0)),
            presence=float(getattr(lm, "presence", 0.0)),
            x_px=x_px, y_px=y_px,
        ))

    world_lms: list[Landmark2D] = []
    pw = getattr(pose_res, "pose_world_landmarks", None) or []
    if pw and best_idx is not None and int(best_idx) < len(pw):
        for lm in pw[int(best_idx)]:
            world_lms.append(Landmark2D(x=float(lm.x), y=float(lm.y), z=float(getattr(lm, "z", 0.0))))

    return int(best_idx), best_metrics, pose_lms, world_lms

# ---- multiprocessing worker state ----
_BACKEND: MediaPipeTasksBackend | None = None
_WORKER_CFG: MediaPipeTasksConfig | None = None

def _init_worker(cfg_dict: dict | None = None):
    global _BACKEND, _WORKER_CFG
    _WORKER_CFG = MediaPipeTasksConfig(**cfg_dict) if cfg_dict else MediaPipeTasksConfig()
    _BACKEND = MediaPipeTasksBackend(cfg=_WORKER_CFG)

def _process_one(
    image_path: Path,
    dataset_dir: Path,
    out_dir: Path,
    debug_overlays: bool,
    save_masks: bool,
    pose_crop_from_mask: bool,
    overlay_thr: float,
    crop_pad_frac: float,
) -> tuple[str, bool]:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = MediaPipeTasksBackend(cfg=_WORKER_CFG or MediaPipeTasksConfig())

    rel = _safe_relpath(image_path, dataset_dir)
    image_id = _image_id_from_rel(rel)
    obs_path = out_dir / "observations" / f"{image_id}.json"
    overlay_path = out_dir / "overlays" / f"{image_id}.png"
    mask_path = out_dir / "masks" / f"{image_id}.png"

    try:
        from PIL import Image, ImageOps
        im = Image.open(str(image_path))
        im = ImageOps.exif_transpose(im)
        rgb = np.array(im.convert('RGB'))
        if rgb is None or rgb.size == 0:
            raise RuntimeError('Image decode produced empty array')
        h, w = rgb.shape[:2]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        seg_obs = None
        crop_bbox: tuple[int,int,int,int] | None = None
        person_mask_u8: np.ndarray | None = None
        seg_meta: dict | None = None
        seg_mode: str | None = None
        cat_mask_np: np.ndarray | None = None

        # Segmentation (used for optional mask saving *and* robust pose cropping)
        raw_mask = None
        raw_kind = None
        uniq_head: list[int | float] = []
        base_mask_u8: np.ndarray | None = None
        alt_mask_u8: np.ndarray | None = None
        base_mode: str | None = None
        alt_mode: str | None = None

        if save_masks or pose_crop_from_mask:
            seg_res = _BACKEND.segment_person(rgb)
            cat_mask = getattr(seg_res, "category_mask", None)
            conf_masks = getattr(seg_res, "confidence_masks", None)

            if cat_mask is not None:
                raw_mask = cat_mask.numpy_view()
                cat_mask_np = raw_mask
                raw_kind = "category_mask"

                # Build two polarity candidates: eq0 and nonzero.
                fg, mode, stats = _choose_foreground_from_category_mask(raw_mask)
                base_mode = mode
                alt_mode = "nonzero" if mode == "eq0" else "eq0"

                base_fg = fg
                alt_fg = (raw_mask != 0) if mode == "eq0" else (raw_mask == 0)

                base_mask_u8 = base_fg.astype(np.uint8) * 255
                alt_mask_u8 = alt_fg.astype(np.uint8) * 255

                # Default (may be overridden after pose scoring).
                person_mask_u8 = base_mask_u8
                seg_mode = base_mode
                seg_meta = {"polarity_mode_initial": base_mode, **stats}

            elif conf_masks:
                raw_mask = conf_masks[0].numpy_view()
                raw_kind = "confidence_mask"
                base_mask_u8 = (np.clip(raw_mask, 0.0, 1.0) * 255.0).astype(np.uint8)

                person_mask_u8 = base_mask_u8
                seg_mode = "confidence"
                seg_meta = {"polarity_mode_initial": "confidence", "area_ratio": float((base_mask_u8 > 127).mean())}

            # Crop bbox for pose (robust against background false positives).
            if pose_crop_from_mask and person_mask_u8 is not None:
                crop_bbox = _bbox_from_mask((person_mask_u8 > 127), pad_frac=crop_pad_frac, w=w, h=h)

            # Unique stats payload for debugging.
            if raw_mask is not None:
                try:
                    uniq = np.unique(raw_mask)
                    uniq_head = [float(x) if str(raw_mask.dtype).startswith("float") else int(x) for x in uniq[:16]]
                except Exception:
                    uniq_head = []

        # Pose (robust selection):
        #   - optionally run on segmentation crop
        #   - optionally try alternate polarity crop
        #   - optionally fall back to full frame
        #   - score candidates using landmark confidence + agreement with person mask
        pose_obs = None
        pose_meta: dict = {"pose_used_crop": bool(crop_bbox is not None)}
        best_pose_lms = None
        best_world_lms = None
        best_bbox = None
        best_source = None
        best_metrics = None
        best_pose_idx = None


        def _try_pose_from_res(
            source: str,
            res,
            *,
            bbox: tuple[int,int,int,int] | None,
            x0: int,
            y0: int,
            rgb_w: int,
            rgb_h: int,
            score_mask_u8: np.ndarray | None,
        ):
            nonlocal best_pose_lms, best_world_lms, best_bbox, best_source, best_metrics, best_pose_idx
            if not (res and getattr(res, "pose_landmarks", None)):
                return
            extracted = _extract_best_pose_from_result(
                res,
                crop_x0=x0, crop_y0=y0, crop_w=rgb_w, crop_h=rgb_h,
                full_w=w, full_h=h,
                person_mask_u8=score_mask_u8,
            )
            if extracted is None:
                return
            pose_idx, metrics, pose_lms, world_lms = extracted
            if best_metrics is None or metrics["score"] > best_metrics["score"]:
                best_pose_lms = pose_lms
                best_world_lms = world_lms
                best_bbox = bbox
                best_source = source
                best_metrics = metrics
                best_pose_idx = pose_idx

        def _try_pose_from_rgb(
            source: str,
            rgb_in: np.ndarray,
            *,
            bbox: tuple[int,int,int,int] | None,
            x0: int,
            y0: int,
            score_mask_u8: np.ndarray | None,
        ):
            if rgb_in is None or rgb_in.size == 0:
                return
            res = _BACKEND.detect_pose(rgb_in)
            cw, ch = rgb_in.shape[1], rgb_in.shape[0]
            _try_pose_from_res(source, res, bbox=bbox, x0=x0, y0=y0, rgb_w=cw, rgb_h=ch, score_mask_u8=score_mask_u8)
        # Candidate 1: segmentation crop if available
        if crop_bbox is not None:
            x0, y0, x1, y1 = crop_bbox
            _try_pose_from_rgb("crop", rgb[y0:y1, x0:x1], bbox=crop_bbox, x0=x0, y0=y0, score_mask_u8=base_mask_u8)


        # Candidate 2: alternate polarity crop (useful if CATEGORY_MASK inference was wrong)
        if alt_mask_u8 is not None:
            alt_bbox = _bbox_from_mask((alt_mask_u8 > 127), pad_frac=crop_pad_frac, w=w, h=h)
            if alt_bbox is not None:
                ax0, ay0, ax1, ay1 = alt_bbox
                _try_pose_from_rgb("alt_crop", rgb[ay0:ay1, ax0:ax1], bbox=alt_bbox, x0=ax0, y0=ay0, score_mask_u8=alt_mask_u8)

        # Candidate 3: full frame (safety net + boots/feet salvage)
        full_res = None

        def _ankles_ok(pose_lms: list[Landmark2D] | None) -> bool:
            if not pose_lms or len(pose_lms) < 29:
                return False
            for idx in (27, 28):  # left/right ankle
                lm = pose_lms[idx]
                if lm.x_px is None or lm.y_px is None:
                    return False
                x = float(lm.x_px); y = float(lm.y_px)
                if x < 0 or x >= float(w) or y < 0 or y >= float(h):
                    return False
                v = float(lm.visibility or 0.0); p = float(lm.presence or 0.0); c = float(min(v if v>0 else p, p if p>0 else v))
                if c < 0.35:
                    return False
            return True

        need_full = (best_metrics is None) or (best_metrics.get("score", 0.0) < 0.55)

        # If the segmentation-derived crop likely truncated the lower body (common with boots /
        # dark shoes), force a full-frame pass so ankles/feet have a chance to be recovered.
        if (not need_full) and (best_bbox is not None):
            try:
                bx0, by0, bx1, by1 = best_bbox
                if (by1 < int(0.97 * h)) and (not _ankles_ok(best_pose_lms)):
                    need_full = True
            except Exception:
                pass

        if need_full:
            full_res = _BACKEND.detect_pose(rgb)
            _try_pose_from_res("full", full_res, bbox=None, x0=0, y0=0, rgb_w=w, rgb_h=h, score_mask_u8=base_mask_u8)
            if alt_mask_u8 is not None:
                _try_pose_from_res("full_alt", full_res, bbox=None, x0=0, y0=0, rgb_w=w, rgb_h=h, score_mask_u8=alt_mask_u8)


        if best_pose_lms is not None:
            pose_obs = PoseObservation(landmarks=best_pose_lms, world_landmarks=best_world_lms or [], names=POSE_LANDMARK_NAMES)
            pose_meta.update({
                "pose_source": best_source,
                "pose_crop_bbox": best_bbox,
                "pose_selected_index": best_pose_idx,
                "pose_score": best_metrics.get("score") if best_metrics else None,
                "pose_mean_conf": best_metrics.get("mean_conf") if best_metrics else None,
                "pose_inmask_ratio": best_metrics.get("inmask_ratio") if best_metrics else None,
                "pose_oob_ratio": best_metrics.get("oob_ratio") if best_metrics else None,
                "pose_geom_penalty": best_metrics.get("geom_penalty") if best_metrics else None,
                "pose_supported_bones": best_metrics.get("supported_bones") if best_metrics else None,
                "pose_support_score": best_metrics.get("support_score") if best_metrics else None,
            })
            crop_bbox = best_bbox  # for overlay rectangle
        else:
            pose_meta.update({"pose_source": None, "pose_score": 0.0})


        
        # Bone-length evidence computed after segmentation polarity selection (mask-gated).

        # Finalize segmentation polarity after pose selection.
        seg_obs = None
        if raw_mask is not None and base_mask_u8 is not None:
            selected_mode = base_mode or seg_mode
            selected_mask = base_mask_u8
            overridden = False
            if alt_mask_u8 is not None and best_source in {"alt_crop", "full_alt"}:
                selected_mask = alt_mask_u8
                selected_mode = alt_mode or selected_mode
                overridden = True

            person_mask_u8 = selected_mask
            seg_mode = selected_mode

            # Save mask after polarity selection (person=white).
            if save_masks and person_mask_u8 is not None:
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(mask_path), person_mask_u8)

            # Stats payload for debugging low-quality real-world inputs.
            seg_obs = SegmentationObservation(
                mask_relpath=str(mask_path.relative_to(out_dir)).replace("\\", "/") if (save_masks and person_mask_u8 is not None) else None,
                mask_kind="selfie_segmenter_square",
                mask_stats={
                    "raw_kind": raw_kind,
                    "raw_dtype": str(getattr(raw_mask, "dtype", None)),
                    "raw_min": float(raw_mask.min()) if hasattr(raw_mask, "min") else None,
                    "raw_max": float(raw_mask.max()) if hasattr(raw_mask, "max") else None,
                    "raw_unique_head": uniq_head,
                    **(seg_meta or {}),
                    "polarity_mode_selected": seg_mode,
                    "polarity_overridden_by_pose": overridden,
                    "selected_area_ratio": float((person_mask_u8 > 127).mean()) if person_mask_u8 is not None else None,
                },
                note="CATEGORY_MASK polarity is selected per-image; may be overridden based on pose/mask agreement.",
            )

            # Ensure crop bbox corresponds to the selected mask for overlay display.
            if pose_crop_from_mask and person_mask_u8 is not None:
                crop_bbox = _bbox_from_mask((person_mask_u8 > 127), pad_frac=crop_pad_frac, w=w, h=h)

        # Bone-length evidence (mask-gated): used later for skeletal proportion solve.
        allowed_idxs = None
        if pose_obs is not None and getattr(pose_obs, "landmarks", None):
            raw_landmarks = pose_obs.landmarks
            raw_world_landmarks = pose_obs.world_landmarks
            bones_px, bones_ratio, bones_conf, ref_bone = _compute_bone_metrics(
                raw_landmarks,
                w=w, h=h,
                names=POSE_LANDMARK_NAMES,
                person_bbox=crop_bbox,
                person_mask_u8=person_mask_u8,
            )
            allowed_idxs = _allowed_pose_indices_from_bones(
                raw_landmarks,
                person_mask_u8=person_mask_u8,
                thr=max(overlay_thr, 0.45),
                w=w, h=h,
                min_len_px=max(12.0, 0.02 * max(w, h)),
            )

            filtered_landmarks: list[Landmark2D] = []
            for idx, lm in enumerate(raw_landmarks):
                if allowed_idxs is None or idx in allowed_idxs:
                    filtered_landmarks.append(lm)
                else:
                    filtered_landmarks.append(
                        Landmark2D(
                            x=float(getattr(lm, "x", 0.0) or 0.0),
                            y=float(getattr(lm, "y", 0.0) or 0.0),
                            z=getattr(lm, "z", 0.0),
                            visibility=0.0,
                            presence=0.0,
                            x_px=None,
                            y_px=None,
                        )
                    )

            filtered_world_landmarks: list[Landmark2D] = []
            for idx, lm in enumerate(raw_world_landmarks):
                if allowed_idxs is None or idx in allowed_idxs:
                    filtered_world_landmarks.append(lm)
                else:
                    filtered_world_landmarks.append(
                        Landmark2D(
                            x=float(getattr(lm, "x", 0.0) or 0.0),
                            y=float(getattr(lm, "y", 0.0) or 0.0),
                            z=float(getattr(lm, "z", 0.0) or 0.0),
                        )
                    )

            pose_obs = PoseObservation(
                landmarks=filtered_landmarks,
                world_landmarks=filtered_world_landmarks,
                names=pose_obs.names,
                bones_px=bones_px,
                bones_ratio=bones_ratio,
                bones_conf=bones_conf,
                raw_landmarks=raw_landmarks,
                raw_world_landmarks=raw_world_landmarks,
            )
            pose_meta.update({
                "bone_ref": ref_bone,
                "bone_valid_count": int(sum(1 for _k, v in bones_conf.items() if float(v) >= 0.25)),
                "pose_allowed_landmarks": int(len(allowed_idxs or set())),
            })
            # Coarse frame-kind classifier (helps downstream weighting; not used to reject).
            if allowed_idxs:
                has_shoulders = (11 in allowed_idxs and 12 in allowed_idxs)
                has_hips = (23 in allowed_idxs and 24 in allowed_idxs)
                has_knees = (25 in allowed_idxs and 26 in allowed_idxs)
                has_ankles = (27 in allowed_idxs and 28 in allowed_idxs)
                has_feet = any(i in allowed_idxs for i in (29, 30, 31, 32))
                if has_shoulders and has_hips and (has_knees or has_ankles):
                    fk = "full_or_mostly_full"
                elif has_shoulders and not has_hips:
                    fk = "upper_body"
                elif has_hips and (has_knees or has_ankles or has_feet) and not has_shoulders:
                    fk = "lower_body"
                elif has_feet and not (has_shoulders or has_hips):
                    fk = "foot_closeup"
                else:
                    fk = "partial"
                pose_meta["frame_kind"] = fk

        # Face
        face_res = _BACKEND.detect_face(rgb)
        face_obs = None
        if face_res and getattr(face_res, "face_landmarks", None):
            flms = face_res.face_landmarks[0]  # first face
            face_lms: list[Landmark2D] = []
            for lm in flms:
                face_lms.append(Landmark2D(
                    x=float(lm.x), y=float(lm.y), z=float(getattr(lm, "z", 0.0)),
                    x_px=float(lm.x * w), y_px=float(lm.y * h),
                ))
            blend = {}
            if getattr(face_res, "face_blendshapes", None):
                # face_blendshapes[0] contains categories
                cats = face_res.face_blendshapes[0]
                for c in cats:
                    name = getattr(c, "category_name", None) or getattr(c, "display_name", None) or "unknown"
                    blend[str(name)] = float(getattr(c, "score", 0.0))
            face_obs = FaceObservation(landmarks=face_lms, blendshapes=blend)

        # Overlays
        if debug_overlays and pose_obs is not None:
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            label = None
            if pose_meta:
                inmask = pose_meta.get('pose_inmask_ratio')
                inmask_s = 'na' if inmask is None else f"{inmask:.2f}"
                keep = pose_meta.get('pose_allowed_landmarks')
                kind = pose_meta.get('frame_kind')
                keep_s = '?' if keep is None else str(int(keep))
                kind_s = '' if kind is None else f" {kind}"
                label = f"{pose_meta.get('pose_source','?')} score={pose_meta.get('pose_score',0.0):.2f} inmask={inmask_s} keep={keep_s}{kind_s}"
            ov = _draw_pose_overlay(bgr, pose_obs.landmarks, thr=overlay_thr, crop_bbox=crop_bbox, label_text=label, person_mask_u8=person_mask_u8, allowed_idxs=allowed_idxs)
            cv2.imwrite(str(overlay_path), ov)

            # Raw overlay for debugging gating effects.
            if getattr(pose_obs, "raw_landmarks", None):
                raw_label = f"raw {label}" if label else "raw"
                raw_overlay_path = overlay_path.with_name(f"{overlay_path.stem}_raw{overlay_path.suffix}")
                raw_ov = _draw_pose_overlay(bgr, pose_obs.raw_landmarks, thr=overlay_thr, crop_bbox=crop_bbox, label_text=raw_label, person_mask_u8=person_mask_u8, allowed_idxs=None)
                cv2.imwrite(str(raw_overlay_path), raw_ov)

        status = Status(ok=True, errors=[])
        obs = Observation(
            image_id=image_id,
            rel_image_path=rel,
            width=w,
            height=h,
            status=status,
            pose=pose_obs,
            face=face_obs,
            segmentation=seg_obs,
            meta={
                **(pose_meta or {}),
                "overlay_threshold": overlay_thr,
            },
        )
        obs_path.parent.mkdir(parents=True, exist_ok=True)
        obs_path.write_text(obs.model_dump_json(indent=2))
        return (image_id, True)

    except Exception as e:
        tb = traceback.format_exc()
        status = Status(ok=False, errors=[f"{type(e).__name__}: {e}"])
        obs = Observation(
            image_id=image_id,
            rel_image_path=rel,
            width=int(locals().get('w', 0) or 0),
            height=int(locals().get('h', 0) or 0),
            status=status,
            meta={"traceback": tb},
        )
        obs_path.parent.mkdir(parents=True, exist_ok=True)
        obs_path.write_text(obs.model_dump_json(indent=2))
        return (image_id, False)

def run_phase1_observations(
    dataset_dir: Path,
    out_dir: Path,
    workers: int = 4,
    debug_overlays: bool = False,
    save_masks: bool = False,
    pose_crop_from_mask: bool = True,
    overlay_thr: float = 0.35,
    crop_pad_frac: float = 0.15,
    mp_cfg: MediaPipeTasksConfig | None = None,
    max_images: int | None = None,
) -> Path:
    images = _iter_images(dataset_dir)
    if max_images is not None:
        images = images[:max_images]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "observations").mkdir(parents=True, exist_ok=True)
    if debug_overlays:
        (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
    if save_masks:
        (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    ok = 0
    failed = 0
    index_path = out_dir / "phase1_index.jsonl"

    # Clear prior index for reruns
    if index_path.exists():
        index_path.unlink()

    # Pass MediaPipe Tasks configuration into each worker so thresholds + model selection
    # are consistent across the run.
    from dataclasses import asdict
    cfg_dict = asdict(mp_cfg or MediaPipeTasksConfig())

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(cfg_dict,)) as ex:
        futures = [
            ex.submit(_process_one, p, dataset_dir, out_dir, debug_overlays, save_masks, pose_crop_from_mask, overlay_thr, crop_pad_frac)
            for p in images
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Phase1 observations"):
            image_id, is_ok = fut.result()
            if is_ok:
                ok += 1
            else:
                failed += 1
            with open(index_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"image_id": image_id, "ok": is_ok}) + "\n")

    summary = {
        "phase": "phase1",
        "dataset_dir": str(dataset_dir.resolve()),
        "total_images": len(images),
        "ok": ok,
        "failed": failed,
        "outputs": {
            "index": str(index_path.name),
            "observations_dir": "observations",
            "overlays_dir": "overlays" if debug_overlays else None,
            "masks_dir": "masks" if save_masks else None,
        },
    }
    (out_dir / "phase1_summary.json").write_text(json.dumps(summary, indent=2))
    return index_path