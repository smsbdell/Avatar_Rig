from __future__ import annotations

from pathlib import Path
import typer
from rich.console import Console

from .stages.phase1_observations import run_phase1_observations
from .stages.phase1_skeleton_solve import run_phase1_skeleton_solve
from .detectors.mediapipe_tasks_backend import MediaPipeTasksConfig

app = typer.Typer(add_completion=False)
console = Console()

@app.command()
def main(
    dataset_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True, help="Dataset directory (can contain nested folders of images)."),
    out: Path = typer.Option(Path("outputs/runs/run1"), "--out", help="Output run directory."),
    workers: int = typer.Option(4, "--workers", min=1, help="Number of worker processes."),
    debug_overlays: bool = typer.Option(False, "--debug-overlays", help="Write overlay images (landmarks + joints) to outputs."),
    save_masks: bool = typer.Option(False, "--save-masks", help="Write segmentation masks to outputs."),
    pose_crop_from_mask: bool = typer.Option(True, "--pose-crop-from-mask/--no-pose-crop-from-mask", help="Use person segmentation to crop the image before pose detection (improves robustness on cluttered photos)."),
    overlay_thr: float = typer.Option(0.35, "--overlay-thr", min=0.0, max=1.0, help="Min presence/visibility to draw pose landmarks in overlays."),
    crop_pad_frac: float = typer.Option(0.15, "--crop-pad-frac", min=0.0, max=0.50, help="Padding fraction applied to the segmentation bounding box before pose detection."),
    pose_model: str = typer.Option("full", "--pose-model", help="Pose model size: lite|full|heavy"),
    pose_profile: str | None = typer.Option(
        None,
        "--pose-profile",
        help=(
            "Pose threshold preset: balanced|strict (overrides --min-pose-* thresholds). "
            "Use strict for cluttered backgrounds or partial-body datasets."
        ),
    ),
    num_poses: int = typer.Option(1, "--num-poses", min=1, help="Max number of poses to detect (kept at 1 for most datasets)."),
    min_pose_det: float = typer.Option(0.30, "--min-pose-det", min=0.0, max=1.0, help="Pose detection confidence threshold."),
    min_pose_presence: float = typer.Option(0.30, "--min-pose-presence", min=0.0, max=1.0, help="Pose presence confidence threshold."),
    min_pose_track: float = typer.Option(0.30, "--min-pose-track", min=0.0, max=1.0, help="Pose tracking confidence threshold (RunningMode.IMAGE uses this as an internal filter)."),
    max_images: int | None = typer.Option(None, "--max-images", help="Optional cap for debugging."),
):
    """Run Phase 1: observations + skeletal proportion solve."""
    out.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]AvatarRig Phase 1[/bold]\nDataset: {dataset_dir}\nOut: {out}\nWorkers: {workers}")
    # Map user-friendly model selector to on-disk task file.
    pose_model = pose_model.strip().lower()
    pose_model_file = {
        "lite": "pose_landmarker_lite.task",
        "full": "pose_landmarker_full.task",
        "heavy": "pose_landmarker_heavy.task",
    }.get(pose_model)
    if pose_model_file is None:
        raise typer.BadParameter("--pose-model must be one of: lite, full, heavy")

    pose_profile = pose_profile.strip().lower() if pose_profile else None
    if pose_profile is not None:
        if pose_profile not in MediaPipeTasksConfig.POSE_PROFILE_PRESETS:
            raise typer.BadParameter("--pose-profile must be one of: balanced, strict")
        thresholds = MediaPipeTasksConfig.apply_pose_profile(pose_profile)
        min_pose_det = thresholds["min_pose_detection_confidence"]
        min_pose_presence = thresholds["min_pose_presence_confidence"]
        min_pose_track = thresholds["min_tracking_confidence"]

    mp_cfg = MediaPipeTasksConfig(
        pose_profile=pose_profile,
        pose_model=pose_model_file,
        num_poses=num_poses,
        min_pose_detection_confidence=min_pose_det,
        min_pose_presence_confidence=min_pose_presence,
        min_tracking_confidence=min_pose_track,
    )

    obs_index = run_phase1_observations(
        dataset_dir=dataset_dir,
        out_dir=out,
        workers=workers,
        debug_overlays=debug_overlays,
        save_masks=save_masks,
        pose_crop_from_mask=pose_crop_from_mask,
        overlay_thr=overlay_thr,
        crop_pad_frac=crop_pad_frac,
        mp_cfg=mp_cfg,
        max_images=max_images,
    )
    run_phase1_skeleton_solve(obs_index_path=obs_index, out_dir=out)

if __name__ == "__main__":
    app()
