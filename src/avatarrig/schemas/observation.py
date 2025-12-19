from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any

class Status(BaseModel):
    ok: bool
    errors: list[str] = Field(default_factory=list)

class Landmark2D(BaseModel):
    x: float
    y: float
    z: float | None = None
    visibility: float | None = None
    presence: float | None = None
    x_px: float | None = None
    y_px: float | None = None

class PoseObservation(BaseModel):
    landmarks: list[Landmark2D] = Field(default_factory=list)
    world_landmarks: list[Landmark2D] = Field(default_factory=list)
    names: list[str] = Field(default_factory=list)

    # Phase 1: we are not solving animation pose; we are extracting joint evidence
    # and limb-length signals for later robust aggregation.
    bones_px: dict[str, float] = Field(default_factory=dict)
    bones_ratio: dict[str, float] = Field(default_factory=dict)
    bones_conf: dict[str, float] = Field(default_factory=dict)

class FaceObservation(BaseModel):
    landmarks: list[Landmark2D] = Field(default_factory=list)
    blendshapes: dict[str, float] = Field(default_factory=dict)

class SegmentationObservation(BaseModel):
    mask_relpath: str | None = None
    mask_kind: str | None = None  # e.g. 'selfie_segmenter_square'
    mask_stats: dict[str, Any] = Field(default_factory=dict)
    note: str | None = None

class Observation(BaseModel):
    schema_version: str = "1.2"
    image_id: str
    rel_image_path: str
    width: int
    height: int
    status: Status
    pose: PoseObservation | None = None
    face: FaceObservation | None = None
    segmentation: SegmentationObservation | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
