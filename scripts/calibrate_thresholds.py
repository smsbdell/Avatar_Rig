from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import typer

app = typer.Typer(add_completion=False, help="Aggregate observation metrics to tune pose thresholds.")


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    pct = min(max(pct, 0.0), 1.0)
    sorted_vals = sorted(values)
    idx = (len(sorted_vals) - 1) * pct
    lower = math.floor(idx)
    upper = math.ceil(idx)
    if lower == upper:
        return float(sorted_vals[int(idx)])
    weight = idx - lower
    return float(sorted_vals[lower] * (1.0 - weight) + sorted_vals[upper] * weight)


def _clamp(val: float, *, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, val)))


def _collect_observation_paths(run_dir: Path) -> list[Path]:
    obs_dir = run_dir / "observations"
    if obs_dir.is_dir():
        return sorted(obs_dir.glob("*.json"))
    return []


def _load_metrics(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text())
    meta = data.get("meta") or {}
    pose_score = meta.get("pose_score")
    inmask = meta.get("pose_inmask_ratio")
    supported = meta.get("pose_supported_bones")
    frame_kind = meta.get("frame_kind")
    return {
        "pose_score": float(pose_score) if pose_score is not None else None,
        "pose_inmask_ratio": float(inmask) if inmask is not None else None,
        "pose_supported_bones": int(supported) if supported is not None else None,
        "frame_kind": str(frame_kind) if frame_kind else None,
    }


def _profile_from_path(path: Path) -> str:
    return path.name


def _summarize_metrics(values: list[float]) -> dict[str, float]:
    clean = [v for v in values if v is not None]
    if not clean:
        return {}
    return {
        "count": len(clean),
        "mean": sum(clean) / len(clean),
        "p10": _percentile(clean, 0.10),
        "p25": _percentile(clean, 0.25),
        "p50": _percentile(clean, 0.50),
        "p75": _percentile(clean, 0.75),
    }


def _recommend_thresholds(scores: list[float], inmask: list[float], bones: list[int]) -> dict[str, float]:
    score_thr = _percentile([v for v in scores if v is not None], 0.15)
    if score_thr is None:
        score_thr = 0.30
    score_thr = _clamp(round(score_thr, 3), lo=0.15, hi=0.90)

    presence_thr = _clamp(round(score_thr - 0.02, 3), lo=0.20, hi=0.90)
    track_thr = _clamp(round(score_thr - 0.04, 3), lo=0.15, hi=0.90)

    inmask_thr = _percentile([v for v in inmask if v is not None], 0.10)
    if inmask_thr is None:
        inmask_thr = 0.35
    inmask_thr = _clamp(round(inmask_thr, 3), lo=0.10, hi=0.95)

    bone_median = _percentile([float(b) for b in bones if b is not None], 0.50)
    overlay_thr = 0.35
    if bone_median is not None:
        if bone_median >= 7:
            overlay_thr = 0.42
        elif bone_median >= 5:
            overlay_thr = 0.38
        elif bone_median <= 2:
            overlay_thr = 0.32
    overlay_thr = _clamp(round(overlay_thr, 3), lo=0.20, hi=0.60)

    min_pose_det = max(score_thr, inmask_thr * 0.8)
    return {
        "min_pose_det": round(min_pose_det, 3),
        "min_pose_presence": presence_thr,
        "min_pose_track": track_thr,
        "overlay_thr": overlay_thr,
    }


def _emit_profile_report(profile: str, stats: dict[str, list[object]], out_lines: list[str]) -> None:
    scores: list[float] = stats.get("pose_score", [])
    inmask: list[float] = stats.get("pose_inmask_ratio", [])
    bones: list[int] = stats.get("pose_supported_bones", [])
    frame_kinds: Counter[str] = Counter(stats.get("frame_kind", []))  # type: ignore[arg-type]

    out_lines.append(f"\n=== Profile: {profile} ===")
    out_lines.append(f"observations: {len(scores)} (pose attempts)")

    summary_score = _summarize_metrics(scores)
    summary_inmask = _summarize_metrics(inmask)
    summary_bones = _summarize_metrics([float(b) for b in bones if b is not None])

    if summary_score:
        out_lines.append("pose_score:")
        out_lines.append(
            f"  mean={summary_score['mean']:.3f} p10={summary_score['p10']:.3f} p25={summary_score['p25']:.3f} "
            f"p50={summary_score['p50']:.3f} p75={summary_score['p75']:.3f}"
        )
    if summary_inmask:
        out_lines.append("pose_inmask_ratio:")
        out_lines.append(
            f"  mean={summary_inmask['mean']:.3f} p10={summary_inmask['p10']:.3f} p25={summary_inmask['p25']:.3f} "
            f"p50={summary_inmask['p50']:.3f} p75={summary_inmask['p75']:.3f}"
        )
    if summary_bones:
        out_lines.append("pose_supported_bones:")
        out_lines.append(
            f"  mean={summary_bones['mean']:.2f} p25={summary_bones['p25']:.2f} p50={summary_bones['p50']:.2f} p75={summary_bones['p75']:.2f}"
        )

    if frame_kinds:
        total_frames = sum(frame_kinds.values())
        top_kinds = ", ".join(
            f"{kind}={count} ({(count/total_frames)*100:.1f}%)"
            for kind, count in frame_kinds.most_common()
        )
        out_lines.append(f"frame_kinds: {top_kinds}")

    rec = _recommend_thresholds(scores, inmask, bones)
    flags = (
        f"--min-pose-det {rec['min_pose_det']:.3f} "
        f"--min-pose-presence {rec['min_pose_presence']:.3f} "
        f"--min-pose-track {rec['min_pose_track']:.3f} "
        f"--overlay-thr {rec['overlay_thr']:.3f}"
    )
    out_lines.append("recommended overrides:")
    out_lines.append(f"  {flags}")


def _aggregate_runs(run_dirs: Iterable[Path]) -> dict[str, dict[str, list[object]]]:
    profiles: dict[str, dict[str, list[object]]] = defaultdict(lambda: defaultdict(list))
    for run_dir in run_dirs:
        obs_paths = _collect_observation_paths(run_dir)
        if not obs_paths:
            continue
        profile = _profile_from_path(run_dir)
        bucket = profiles[profile]
        for obs_path in obs_paths:
            metrics = _load_metrics(obs_path)
            for key, value in metrics.items():
                if value is None:
                    continue
                bucket[key].append(value)
    return profiles


@app.command()
def calibrate(
    root: Path = typer.Argument(Path("outputs/runs"), exists=True, help="Root directory containing run folders."),
    report_path: Path | None = typer.Option(None, "--report", help="Optional path to save the text report."),
):
    """Aggregate observation metrics from one or more runs and propose thresholds."""
    if (root / "observations").is_dir():
        run_dirs = [root]
    else:
        run_dirs = [p for p in root.iterdir() if p.is_dir() and (p / "observations").is_dir()]

    if not run_dirs:
        typer.echo(f"No observation folders found under {root}.")
        raise typer.Exit(code=1)

    profiles = _aggregate_runs(run_dirs)
    if not profiles:
        typer.echo("No metrics found in observations.")
        raise typer.Exit(code=1)

    lines: list[str] = []
    lines.append("AvatarRig threshold calibration")
    lines.append(f"scanned runs: {', '.join(sorted(p.name for p in run_dirs))}")

    for profile, stats in sorted(profiles.items()):
        _emit_profile_report(profile, stats, lines)

    report = "\n".join(lines)
    typer.echo(report)

    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report)
        typer.echo(f"Saved report to {report_path}")


if __name__ == "__main__":
    app()
