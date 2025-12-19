from __future__ import annotations

from pathlib import Path
import json
import math
from statistics import median

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
    obs_dir = out_dir / "observations"
    ratios_by_key: dict[str, list[float]] = {}

    used = 0
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

    # Median-aggregate: robust under high-noise, high-volume datasets.
    est = {k: median(vs) for k, vs in ratios_by_key.items() if len(vs) >= 50}  # require some support
    meta = {k: {"n": len(vs), "median": median(vs)} for k, vs in ratios_by_key.items()}

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
    return out_path
