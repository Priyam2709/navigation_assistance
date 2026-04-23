from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Annotation:
    class_name: str
    bbox_xywh_abs: list[float]
    ignore: bool = False
    source_class: str | None = None
    track_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetRecord:
    record_id: str
    source_dataset: str
    split: str
    image_path: str
    image_relpath: str
    width: int | None
    height: int | None
    original_id: str
    video_id: str | None = None
    frame_index: int | None = None
    track_key: str | None = None
    annotations: list[Annotation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["annotations"] = [annotation.to_dict() for annotation in self.annotations]
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetRecord":
        annotations = [Annotation(**item) for item in data.get("annotations", [])]
        payload = {key: value for key, value in data.items() if key != "annotations"}
        return cls(annotations=annotations, **payload)
