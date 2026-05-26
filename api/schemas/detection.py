from pydantic import BaseModel
from typing import List, Optional


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionResult(BaseModel):
    class_name: str
    confidence: float
    bbox: BoundingBox
    track_id: Optional[int] = None


class PPEStatusSchema(BaseModel):
    track_id: int
    has_helmet: bool
    has_vest: bool
    has_glasses: bool
    is_compliant: bool
    violations: List[str]


class PersonSchema(BaseModel):
    track_id: Optional[int]
    bbox: List[float]
    bbox_px: List[int]
    confidence: float
    ppe_status: Optional[PPEStatusSchema]


class ViolationSchema(BaseModel):
    track_id: int
    violations: List[str]
    frame: int


class DetectionResponse(BaseModel):
    status: str
    detections: List[DetectionResult]


class FullPipelineResponse(BaseModel):
    status: str
    person_detections: List[DetectionResult]
    ppe_detections: List[DetectionResult]
    annotated_image_base64: Optional[str] = None