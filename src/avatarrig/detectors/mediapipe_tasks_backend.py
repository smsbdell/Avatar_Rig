from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import mediapipe as mp

from ..util.download import download_if_missing
from ..util.repo import find_repo_root

POSE_LANDMARK_NAMES = [
    "nose",
    "left_eye_inner","left_eye","left_eye_outer",
    "right_eye_inner","right_eye","right_eye_outer",
    "left_ear","right_ear",
    "mouth_left","mouth_right",
    "left_shoulder","right_shoulder",
    "left_elbow","right_elbow",
    "left_wrist","right_wrist",
    "left_pinky","right_pinky",
    "left_index","right_index",
    "left_thumb","right_thumb",
    "left_hip","right_hip",
    "left_knee","right_knee",
    "left_ankle","right_ankle",
    "left_heel","right_heel",
    "left_foot_index","right_foot_index",
]

POSE_MODEL_URL_LITE = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
POSE_MODEL_URL_FULL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
POSE_MODEL_URL_HEAVY = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
SELFIE_SEG_URL_SQUARE = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"

@dataclass(frozen=True)
class MediaPipeTasksConfig:
    POSE_PROFILE_PRESETS = {
        "balanced": {
            "min_pose_detection_confidence": 0.30,
            "min_pose_presence_confidence": 0.30,
            "min_tracking_confidence": 0.30,
        },
        "strict": {
            "min_pose_detection_confidence": 0.60,
            "min_pose_presence_confidence": 0.60,
            "min_tracking_confidence": 0.60,
        },
    }

    pose_profile: str | None = None
    pose_model: str = "pose_landmarker_full.task"
    face_model: str = "face_landmarker.task"
    seg_model: str = "selfie_segmenter.tflite"
    output_pose_segmentation: bool = False
    output_face_blendshapes: bool = True
    num_faces: int = 1

    # PoseLandmarkerOptions (tunable for low-fidelity, uncontrolled datasets)
    num_poses: int = 1
    min_pose_detection_confidence: float = 0.30
    min_pose_presence_confidence: float = 0.30
    min_tracking_confidence: float = 0.30

    def __post_init__(self):
        if self.pose_profile is None:
            return
        profile_key = self.pose_profile.strip().lower()
        if profile_key not in self.POSE_PROFILE_PRESETS:
            raise ValueError(f"Unknown pose profile '{self.pose_profile}' (expected one of: {', '.join(self.POSE_PROFILE_PRESETS)})")
        object.__setattr__(self, "pose_profile", profile_key)

    @classmethod
    def apply_pose_profile(cls, pose_profile: str) -> dict[str, float]:
        profile_key = pose_profile.strip().lower()
        if profile_key not in cls.POSE_PROFILE_PRESETS:
            raise ValueError(
                f"Unknown pose profile '{pose_profile}' (expected one of: {', '.join(cls.POSE_PROFILE_PRESETS)})"
            )
        return cls.POSE_PROFILE_PRESETS[profile_key]

class MediaPipeTasksBackend:
    def __init__(self, cfg: MediaPipeTasksConfig | None = None, repo_root: Path | None = None):
        self.cfg = cfg or MediaPipeTasksConfig()
        self.repo_root = repo_root or find_repo_root()

        models_dir = self.repo_root / "inputs" / "models" / "mediapipe"
        pose_path = models_dir / self.cfg.pose_model
        face_path = models_dir / self.cfg.face_model
        seg_path  = models_dir / self.cfg.seg_model

        # Download defaults if missing.
        if self.cfg.pose_model == "pose_landmarker_lite.task":
            download_if_missing(POSE_MODEL_URL_LITE, pose_path)
        elif self.cfg.pose_model == "pose_landmarker_full.task":
            download_if_missing(POSE_MODEL_URL_FULL, pose_path)
        elif self.cfg.pose_model == "pose_landmarker_heavy.task":
            download_if_missing(POSE_MODEL_URL_HEAVY, pose_path)
        if self.cfg.face_model == "face_landmarker.task":
            download_if_missing(FACE_MODEL_URL, face_path)
        if self.cfg.seg_model == "selfie_segmenter.tflite":
            download_if_missing(SELFIE_SEG_URL_SQUARE, seg_path)

        BaseOptions = mp.tasks.BaseOptions
        vision = mp.tasks.vision

        pose_opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(pose_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=self.cfg.num_poses,
            min_pose_detection_confidence=self.cfg.min_pose_detection_confidence,
            min_pose_presence_confidence=self.cfg.min_pose_presence_confidence,
            min_tracking_confidence=self.cfg.min_tracking_confidence,
            output_segmentation_masks=self.cfg.output_pose_segmentation,
        )
        self._pose = vision.PoseLandmarker.create_from_options(pose_opts)

        face_opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(face_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=self.cfg.num_faces,
            output_face_blendshapes=self.cfg.output_face_blendshapes,
        )
        self._face = vision.FaceLandmarker.create_from_options(face_opts)

        seg_opts = vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=str(seg_path)),
            running_mode=vision.RunningMode.IMAGE,
            output_category_mask=True,
            output_confidence_masks=False,
        )
        self._seg = vision.ImageSegmenter.create_from_options(seg_opts)

    @staticmethod
    def _to_mp_image(rgb: np.ndarray) -> mp.Image:
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    def detect_pose(self, rgb: np.ndarray):
        return self._pose.detect(self._to_mp_image(rgb))

    def detect_face(self, rgb: np.ndarray):
        return self._face.detect(self._to_mp_image(rgb))

    def segment_person(self, rgb: np.ndarray):
        return self._seg.segment(self._to_mp_image(rgb))
