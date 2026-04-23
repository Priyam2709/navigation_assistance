from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs"
TEMPLATES_ROOT = PROJECT_ROOT / "templates"
STATIC_ROOT = PROJECT_ROOT / "static"
RUNS_ROOT = PROJECT_ROOT / "runs"


@dataclass(frozen=True)
class InferenceConfig:
    conf: float
    imgsz: int
    device: str
    max_frames: int
    min_track_observations: int


@dataclass(frozen=True)
class AppConfig:
    app_name: str
    station_config_path: Path
    weights_path: Path
    demo_video_path: Path
    storage_root: Path
    upload_root: Path
    inference: InferenceConfig


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def default_weights_path() -> Path:
    candidates = [
        PROJECT_ROOT / "runs" / "detect" / "crossroad_baseline_smoke" / "weights" / "best.pt",
        PROJECT_ROOT / "yolo11n.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def load_app_config(path: Path | None = None) -> AppConfig:
    config_path = path or (CONFIG_ROOT / "app.yaml")
    payload = _load_yaml(config_path)
    inference_payload = payload.get("inference", {})

    weights_path = _resolve_path(payload.get("weights", str(default_weights_path())))
    if not weights_path.exists():
        weights_path = default_weights_path()

    return AppConfig(
        app_name=str(payload.get("app_name", "Navigation Assistance System")),
        station_config_path=_resolve_path(payload.get("station_config", "configs/station_demo.json")),
        weights_path=weights_path,
        demo_video_path=_resolve_path(payload.get("demo_video", "runs/demo_video/crossroad_demo.mp4")),
        storage_root=_resolve_path(payload.get("storage_root", "runs/app_state")),
        upload_root=_resolve_path(payload.get("upload_root", "runs/uploads")),
        inference=InferenceConfig(
            conf=float(inference_payload.get("conf", 0.25)),
            imgsz=int(inference_payload.get("imgsz", 512)),
            device=str(inference_payload.get("device", "0")),
            max_frames=int(inference_payload.get("max_frames", 180)),
            min_track_observations=int(inference_payload.get("min_track_observations", 2)),
        ),
    )

