from __future__ import annotations

from pathlib import Path
import json
import math
from statistics import median

import cv2
import numpy as np

from ..schemas.observation import Observation

# Pose landmark indices (BlazePose)
IDX = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
}

def _dist2(a, b) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])


def _lm_conf(lm) -> float:
    v = float(getattr(lm, 'visibility', 0.0) or 0.0)
    p = float(getattr(lm, 'presence', 0.0) or 0.0)
    if p <= 0.0:
        p = v
    if v <= 0.0:
        v = p
    return float(min(v, p))


def _usable_for_skeleton(obs: Observation) -> bool:
    """Heuristic filter to reduce outliers from partial-body / close-up frames."""
    m = getattr(obs, "meta", None) or {}
    try:
        score = float(m.get("pose_score", 0.0) or 0.0)
        if score < 0.70:
            return False
        inmask = m.get("pose_inmask_ratio", None)
        if inmask is not None and float(inmask) < 0.60:
            return False
        oob = m.get("pose_oob_ratio", None)
        if oob is not None and float(oob) > 0.15:
            return False
        gp = m.get("pose_geom_penalty", None)
        if gp is not None and float(gp) < 0.50:
            return False
        sb = m.get("pose_supported_bones", None)
        if sb is not None and int(sb) < 3:
            return False
    except Exception:
        # If meta is missing or malformed, fall back to landmark-only gating.
        pass
    return True

def _get_xy(obs: Observation, name: str, *, thr: float = 0.35):
    if not obs.pose or not obs.pose.landmarks:
        return None
    i = IDX[name]
    lm = obs.pose.landmarks[i]
    if _lm_conf(lm) < thr:
        return None
    # Use normalized coords (invariant to resolution)
    return (lm.x, lm.y)

def _mid(a, b):
    return ((a[0]+b[0])*0.5, (a[1]+b[1])*0.5)

def _ratios_from_obs(obs: Observation) -> dict[str, float] | None:
    if not _usable_for_skeleton(obs):
        return None
    ls = _get_xy(obs, "left_shoulder")
    rs = _get_xy(obs, "right_shoulder")
    lh = _get_xy(obs, "left_hip")
    rh = _get_xy(obs, "right_hip")
    if not (ls and rs and lh and rh):
        return None

    shoulder_w = _dist2(ls, rs)
    hip_w = _dist2(lh, rh)
    torso = _dist2(_mid(ls, rs), _mid(lh, rh))

    # Require a non-trivial shoulder width to act as our arbitrary but consistent scale.
    if shoulder_w < 1e-6:
        return None

    def seg(a_name, b_name):
        a = _get_xy(obs, a_name)
        b = _get_xy(obs, b_name)
        if not (a and b):
            return None
        return _dist2(a, b)

    upper_arm_L = seg("left_shoulder", "left_elbow")
    lower_arm_L = seg("left_elbow", "left_wrist")
    upper_arm_R = seg("right_shoulder", "right_elbow")
    lower_arm_R = seg("right_elbow", "right_wrist")

    upper_leg_L = seg("left_hip", "left_knee")
    lower_leg_L = seg("left_knee", "left_ankle")
    upper_leg_R = seg("right_hip", "right_knee")
    lower_leg_R = seg("right_knee", "right_ankle")

    out = {
        "shoulder_width": 1.0,
        "hip_width": hip_w / shoulder_w,
        "torso_length": torso / shoulder_w,
    }
    for k, v in {
        "upper_arm_L": upper_arm_L, "lower_arm_L": lower_arm_L,
        "upper_arm_R": upper_arm_R, "lower_arm_R": lower_arm_R,
        "upper_leg_L": upper_leg_L, "lower_leg_L": lower_leg_L,
        "upper_leg_R": upper_leg_R, "lower_leg_R": lower_leg_R,
    }.items():
        if v is not None:
            out[k] = v / shoulder_w

    # Derived "wingspan" estimate from bone lengths; avoids dependence on a single pose.
    arm_L = (out.get("upper_arm_L", None), out.get("lower_arm_L", None))
    arm_R = (out.get("upper_arm_R", None), out.get("lower_arm_R", None))
    if all(x is not None for x in arm_L + arm_R):
        wingspan = (arm_L[0] + arm_L[1]) + (arm_R[0] + arm_R[1]) + 1.0  # + shoulder width
        out["wingspan_est"] = wingspan
    return out

def run_phase1_skeleton_solve(obs_index_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    obs_dir = out_dir / "observations"
    ratios_by_key: dict[str, list[float]] = {}

    used = 0

    def _current_estimate() -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        est = {k: median(vs) for k, vs in ratios_by_key.items() if len(vs) >= 50}  # require some support
        meta = {k: {"n": len(vs), "median": median(vs)} for k, vs in ratios_by_key.items()}
        return est, meta

    def _write_outputs(est: dict[str, float], meta: dict[str, dict[str, float]]) -> Path:
        out = {
            "schema_version": "1.0",
            "phase": "phase1_skeleton_solve",
            "used_observations": used,
            "estimates": est,
            "stats": meta,
            "scale_basis": "shoulder_width == 1.0 (normalized image coordinate space)",
            "notes": [
                "These are first-pass proportions. Noise/outliers are expected with real-world photos.",
                "We use median aggregation across many frames to make the estimate robust.",
            ],
        }
        out_path = out_dir / "skeleton_estimate.json"
        out_path.write_text(json.dumps(out, indent=2))
        _export_tpose_preview(est, out_dir / "skeleton_estimate.png")
        return out_path

    est, meta = _current_estimate()
    out_path = _write_outputs(est, meta)

    with open(obs_index_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if not row.get("ok"):
                continue
            obs_path = obs_dir / f"{row['image_id']}.json"
            if not obs_path.exists():
                continue
            obs = Observation.model_validate_json(obs_path.read_text(encoding="utf-8"))
            r = _ratios_from_obs(obs)
            if not r:
                continue
            used += 1
            for k, v in r.items():
                ratios_by_key.setdefault(k, []).append(float(v))

            est, meta = _current_estimate()
            out_path = _write_outputs(est, meta)
    return out_path


def _export_tpose_preview(est: dict[str, float], out_path: Path, *, size: int = 800) -> None:
    """Render a simple T-pose visualization of the median joint rig.

    The pose is normalized such that shoulder_width == 1.0; all other lengths
    are stored as ratios in the estimate. The resulting image overwrites any
    existing file so downstream consumers always see the latest solve.
    """

    def _seg_len(primary: float | None, fallback: float) -> float:
        return float(primary) if primary is not None else fallback

    shoulder_w = _seg_len(est.get("shoulder_width"), 1.0)
    hip_w = _seg_len(est.get("hip_width"), 0.7) * shoulder_w
    torso_len = _seg_len(est.get("torso_length"), 1.0) * shoulder_w

    wingspan = est.get("wingspan_est")
    def _arm_lengths(prefix: str) -> tuple[float, float]:
        upper = est.get(f"upper_arm_{prefix}")
        lower = est.get(f"lower_arm_{prefix}")
        if (upper is None or lower is None) and wingspan is not None:
            half = max(float(wingspan) - shoulder_w, 0.0) * 0.5
            if upper is None and lower is None and half > 0.0:
                upper = half * 0.55
                lower = half * 0.45
            elif upper is None and lower is not None:
                upper = max(half - lower, lower * 0.8)
            elif lower is None and upper is not None:
                lower = max(half - upper, upper * 0.8)
        return _seg_len(upper, 0.6 * shoulder_w), _seg_len(lower, 0.4 * shoulder_w)

    def _leg_lengths(prefix: str) -> tuple[float, float]:
        upper = _seg_len(est.get(f"upper_leg_{prefix}"), 1.2 * shoulder_w)
        lower = _seg_len(est.get(f"lower_leg_{prefix}"), 1.2 * shoulder_w)
        return upper, lower

    ua_L, la_L = _arm_lengths("L")
    ua_R, la_R = _arm_lengths("R")
    ul_L, ll_L = _leg_lengths("L")
    ul_R, ll_R = _leg_lengths("R")

    joints = {}
    joints["left_shoulder"] = (-0.5 * shoulder_w, 0.0)
    joints["right_shoulder"] = (0.5 * shoulder_w, 0.0)
    joints["left_elbow"] = (joints["left_shoulder"][0] - ua_L, 0.0)
    joints["right_elbow"] = (joints["right_shoulder"][0] + ua_R, 0.0)
    joints["left_wrist"] = (joints["left_elbow"][0] - la_L, 0.0)
    joints["right_wrist"] = (joints["right_elbow"][0] + la_R, 0.0)

    hip_y = torso_len
    joints["left_hip"] = (-0.5 * hip_w, hip_y)
    joints["right_hip"] = (0.5 * hip_w, hip_y)
    joints["left_knee"] = (joints["left_hip"][0], hip_y + ul_L)
    joints["right_knee"] = (joints["right_hip"][0], hip_y + ul_R)
    joints["left_ankle"] = (joints["left_knee"][0], joints["left_knee"][1] + ll_L)
    joints["right_ankle"] = (joints["right_knee"][0], joints["right_knee"][1] + ll_R)

    pts = np.array(list(joints.values()), dtype=float)
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    center = (min_xy + max_xy) * 0.5
    span = float(np.max(max_xy - min_xy))
    span = span if span > 1e-3 else 1.0
    scale = (size * 0.8) / span

    def to_px(pt: tuple[float, float]) -> tuple[int, int]:
        x, y = pt
        px = int(round(size * 0.5 + (x - center[0]) * scale))
        py = int(round(size * 0.5 + (y - center[1]) * scale))
        return px, py

    img = np.zeros((size, size, 3), dtype=np.uint8)
    bones = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ]

    for a, b in bones:
        cv2.line(img, to_px(joints[a]), to_px(joints[b]), (0, 255, 255), 4, cv2.LINE_AA)
    for p in joints.values():
        cv2.circle(img, to_px(p), 8, (0, 165, 255), -1, lineType=cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
