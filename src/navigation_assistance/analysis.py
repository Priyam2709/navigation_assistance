from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import cv2

from .tracker import AssistanceTracker, FrameDetection, SPECIAL_CLASSES, bbox_iou

FRAME_FONT = cv2.FONT_HERSHEY_SIMPLEX


@lru_cache(maxsize=2)
def _load_model(weights_path: str):
    from ultralytics import YOLO

    return YOLO(weights_path)


def _to_web_path(project_root: Path, path: Path) -> str:
    return "/" + str(path.resolve().relative_to(project_root.resolve())).replace("\\", "/")


def _special_color(label: str) -> tuple[int, int, int]:
    palette = {
        "wheelchair_user": (52, 152, 219),
        "walker_user": (241, 196, 15),
        "crutch_user": (231, 76, 60),
        "cane_user": (46, 204, 113),
    }
    return palette.get(label, (255, 255, 255))


def _extract_special_detections(result, frame_width: int, frame_height: int) -> list[FrameDetection]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    names = {int(key): value for key, value in result.names.items()} if isinstance(result.names, dict) else {}
    candidates: list[FrameDetection] = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        class_name = names.get(class_id, str(class_id))
        if class_name not in SPECIAL_CLASSES:
            continue
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        center_x_norm = ((x1 + x2) / 2.0) / frame_width
        center_y_norm = ((y1 + y2) / 2.0) / frame_height
        candidates.append(
            FrameDetection(
                class_name=class_name,
                confidence=confidence,
                bbox_xyxy=[x1, y1, x2, y2],
                center_norm=(center_x_norm, center_y_norm),
            )
        )

    # Class-aware dedupe to reduce overlapping aid labels on the same subject.
    deduped: list[FrameDetection] = []
    for detection in sorted(candidates, key=lambda item: item.confidence, reverse=True):
        if any(bbox_iou(detection.bbox_xyxy, kept.bbox_xyxy) > 0.65 for kept in deduped):
            continue
        deduped.append(detection)
    return deduped


def _draw_zone_overlay(frame, station, frame_width: int, frame_height: int) -> None:
    overlay = frame.copy()
    for zone in station.frame_zones:
        x_start, x_end = zone["x_range"]
        y_start, y_end = zone["y_range"]
        left = int(x_start * frame_width)
        right = int(x_end * frame_width)
        top = int(y_start * frame_height)
        bottom = int(y_end * frame_height)
        cv2.rectangle(overlay, (left, top), (right, bottom), (0, 40, 60), -1)
        cv2.putText(
            overlay,
            zone["label"],
            (left + 6, min(bottom - 8, top + 22)),
            FRAME_FONT,
            0.45,
            (220, 240, 245),
            1,
            cv2.LINE_AA,
        )
    cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)


def _draw_track(frame, track, station, frame_width: int, frame_height: int) -> None:
    observation = track.observations[-1]
    x1, y1, x2, y2 = [int(value) for value in observation["bbox_xyxy"]]
    color = _special_color(track.dominant_label)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"T{track.track_id} {track.dominant_label} {observation['zone_label']}"
    text_y = max(18, y1 - 10)
    cv2.putText(frame, label, (x1, text_y), FRAME_FONT, 0.55, color, 2, cv2.LINE_AA)

    center_x = int(observation["center_norm"][0] * frame_width)
    center_y = int(observation["center_norm"][1] * frame_height)
    cv2.circle(frame, (center_x, center_y), 4, color, -1)


def analyze_video(
    *,
    project_root: Path,
    source_path: Path,
    output_root: Path,
    station,
    weights_path: Path,
    conf: float,
    imgsz: int,
    device: str,
    max_frames: int,
    min_track_observations: int,
) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    analysis_id = f"analysis-{uuid4().hex[:10]}"
    analysis_dir = output_root / analysis_id
    analysis_dir.mkdir(parents=True, exist_ok=True)

    annotated_video_path = analysis_dir / "annotated.mp4"
    metadata_path = analysis_dir / "analysis.json"
    crops_dir = analysis_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model(str(weights_path.resolve()))
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 8.0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    writer = cv2.VideoWriter(
        str(annotated_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )

    tracker = AssistanceTracker()
    frame_index = 0
    best_confidences: dict[int, float] = {}

    try:
        while frame_index < max_frames:
            success, frame = capture.read()
            if not success:
                break
            result = model.predict(frame, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
            detections = _extract_special_detections(result, frame_width, frame_height)
            matched_tracks = tracker.update(frame_index, detections, station.assign_zone)

            _draw_zone_overlay(frame, station, frame_width, frame_height)
            cv2.putText(
                frame,
                f"{station.station_name} | Frame {frame_index}",
                (18, frame_height - 18),
                FRAME_FONT,
                0.7,
                (248, 248, 248),
                2,
                cv2.LINE_AA,
            )
            for track in matched_tracks:
                _draw_track(frame, track, station, frame_width, frame_height)
                # Capture visual proof if confidence improves
                obs = track.observations[-1]
                if obs["confidence"] > best_confidences.get(track.track_id, 0.0):
                    best_confidences[track.track_id] = obs["confidence"]
                    x1, y1, x2, y2 = [int(v) for v in obs["bbox_xyxy"]]
                    x1, y1 = max(0, x1 - 15), max(0, y1 - 15)
                    x2, y2 = min(frame_width, x2 + 15), min(frame_height, y2 + 15)
                    crop_img = frame[y1:y2, x1:x2]
                    if crop_img.size > 0:
                        crop_path = crops_dir / f"track_{track.track_id}.jpg"
                        cv2.imwrite(str(crop_path), crop_img)

            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    tracker.finalize()
    passengers = tracker.export_tracks(min_observations=min_track_observations)

    for passenger in passengers:
        crop_path = crops_dir / f"track_{passenger['track_id']}.jpg"
        if crop_path.exists():
            passenger["crop_url"] = _to_web_path(project_root, crop_path)
        else:
            passenger["crop_url"] = ""

    analysis_payload = {
        "analysis_id": analysis_id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "station_name": station.station_name,
        "source_video_path": str(source_path.resolve()),
        "source_video_url": _to_web_path(project_root, source_path),
        "annotated_video_path": str(annotated_video_path.resolve()),
        "annotated_video_url": _to_web_path(project_root, annotated_video_path),
        "weights_path": str(weights_path.resolve()),
        "frames_analyzed": frame_index,
        "fps": fps,
        "duration_seconds": round(frame_index / fps, 2) if fps else 0.0,
        "frame_size": {"width": frame_width, "height": frame_height},
        "special_needs_count": len(passengers),
        "passengers": passengers,
        "destinations": station.get_destinations(),
        "station_nodes": station.node_catalog(),
        "station_edges": station.edge_catalog(),
    }
    metadata_path.write_text(json.dumps(analysis_payload, indent=2), encoding="utf-8")
    return analysis_payload
