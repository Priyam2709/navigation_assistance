from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any


SPECIAL_CLASSES = {"wheelchair_user", "walker_user", "crutch_user", "cane_user"}


@dataclass
class FrameDetection:
    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    center_norm: tuple[float, float]


@dataclass
class Track:
    track_id: int
    current_bbox: list[float]
    last_frame: int
    missed_frames: int = 0
    observations: list[dict[str, Any]] = field(default_factory=list)
    label_votes: Counter = field(default_factory=Counter)
    max_confidence: float = 0.0

    def update(self, frame_index: int, detection: FrameDetection, zone_id: str, zone_label: str) -> None:
        self.current_bbox = detection.bbox_xyxy
        self.last_frame = frame_index
        self.missed_frames = 0
        self.label_votes[detection.class_name] += 1
        self.max_confidence = max(self.max_confidence, detection.confidence)
        self.observations.append(
            {
                "frame_index": frame_index,
                "class_name": detection.class_name,
                "confidence": round(detection.confidence, 5),
                "bbox_xyxy": [round(value, 2) for value in detection.bbox_xyxy],
                "center_norm": [round(detection.center_norm[0], 5), round(detection.center_norm[1], 5)],
                "zone_id": zone_id,
                "zone_label": zone_label,
            }
        )

    @property
    def observation_count(self) -> int:
        return len(self.observations)

    @property
    def dominant_label(self) -> str:
        special_votes = Counter({label: count for label, count in self.label_votes.items() if label in SPECIAL_CLASSES})
        if special_votes:
            return special_votes.most_common(1)[0][0]
        return self.label_votes.most_common(1)[0][0] if self.label_votes else "unknown"


def bbox_iou(left_bbox: list[float], right_bbox: list[float]) -> float:
    left_x1, left_y1, left_x2, left_y2 = left_bbox
    right_x1, right_y1, right_x2, right_y2 = right_bbox
    inter_x1 = max(left_x1, right_x1)
    inter_y1 = max(left_y1, right_y1)
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height
    left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
    right_area = max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
    denom = left_area + right_area - inter_area
    return inter_area / denom if denom else 0.0


def _center_distance(left_center: tuple[float, float], right_center: tuple[float, float]) -> float:
    return ((left_center[0] - right_center[0]) ** 2 + (left_center[1] - right_center[1]) ** 2) ** 0.5


class AssistanceTracker:
    def __init__(self, max_missed_frames: int = 12, min_iou: float = 0.15, max_center_distance: float = 0.12) -> None:
        self.max_missed_frames = max_missed_frames
        self.min_iou = min_iou
        self.max_center_distance = max_center_distance
        self._next_track_id = 1
        self.active_tracks: dict[int, Track] = {}
        self.completed_tracks: dict[int, Track] = {}

    def update(self, frame_index: int, detections: list[FrameDetection], zone_resolver) -> list[Track]:
        candidates: list[tuple[float, int, int]] = []
        active_track_ids = list(self.active_tracks)
        for track_id in active_track_ids:
            track = self.active_tracks[track_id]
            previous_center = track.observations[-1]["center_norm"] if track.observations else (0.0, 0.0)
            previous_center_pair = (float(previous_center[0]), float(previous_center[1]))
            for detection_index, detection in enumerate(detections):
                iou = bbox_iou(track.current_bbox, detection.bbox_xyxy)
                distance = _center_distance(previous_center_pair, detection.center_norm)
                if iou >= self.min_iou or distance <= self.max_center_distance:
                    class_bonus = 0.2 if detection.class_name == track.dominant_label else 0.0
                    score = iou + class_bonus + (0.25 - min(distance, 0.25))
                    candidates.append((score, track_id, detection_index))

        candidates.sort(reverse=True)
        assigned_tracks: set[int] = set()
        assigned_detections: set[int] = set()
        matched_tracks: list[Track] = []

        for _, track_id, detection_index in candidates:
            if track_id in assigned_tracks or detection_index in assigned_detections:
                continue
            track = self.active_tracks[track_id]
            detection = detections[detection_index]
            zone_match = zone_resolver(*detection.center_norm)
            track.update(frame_index, detection, zone_match.zone_id, zone_match.label)
            assigned_tracks.add(track_id)
            assigned_detections.add(detection_index)
            matched_tracks.append(track)

        for track_id in list(self.active_tracks):
            if track_id not in assigned_tracks:
                track = self.active_tracks[track_id]
                track.missed_frames += 1
                if track.missed_frames > self.max_missed_frames:
                    self.completed_tracks[track_id] = self.active_tracks.pop(track_id)

        for detection_index, detection in enumerate(detections):
            if detection_index in assigned_detections:
                continue
            zone_match = zone_resolver(*detection.center_norm)
            track = Track(
                track_id=self._next_track_id,
                current_bbox=detection.bbox_xyxy,
                last_frame=frame_index,
            )
            track.update(frame_index, detection, zone_match.zone_id, zone_match.label)
            self.active_tracks[self._next_track_id] = track
            self._next_track_id += 1
            matched_tracks.append(track)

        return matched_tracks

    def finalize(self) -> None:
        for track_id in list(self.active_tracks):
            self.completed_tracks[track_id] = self.active_tracks.pop(track_id)

    def export_tracks(self, min_observations: int) -> list[dict[str, Any]]:
        all_tracks = list(self.completed_tracks.values()) + list(self.active_tracks.values())
        exported: list[dict[str, Any]] = []
        for track in all_tracks:
            if track.observation_count < min_observations:
                continue
            if track.dominant_label not in SPECIAL_CLASSES:
                continue
            first_observation = track.observations[0]
            last_observation = track.observations[-1]
            zone_path = []
            for observation in track.observations:
                if not zone_path or zone_path[-1]["zone_id"] != observation["zone_id"]:
                    zone_path.append({"zone_id": observation["zone_id"], "zone_label": observation["zone_label"]})
            exported.append(
                {
                    "track_id": track.track_id,
                    "dominant_label": track.dominant_label,
                    "label_votes": dict(track.label_votes),
                    "max_confidence": round(track.max_confidence, 5),
                    "first_frame": first_observation["frame_index"],
                    "last_frame": last_observation["frame_index"],
                    "observation_count": track.observation_count,
                    "current_zone_id": last_observation["zone_id"],
                    "current_zone_label": last_observation["zone_label"],
                    "latest_bbox_xyxy": last_observation["bbox_xyxy"],
                    "observations": track.observations,
                    "zone_path": zone_path,
                }
            )
        exported.sort(key=lambda item: (item["current_zone_id"], item["track_id"]))
        return exported
