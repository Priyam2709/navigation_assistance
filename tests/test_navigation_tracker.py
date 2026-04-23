from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from navigation_assistance.station import DemoStation
from navigation_assistance.tracker import AssistanceTracker, FrameDetection


class TrackerTests(unittest.TestCase):
    def test_tracker_keeps_same_id_for_nearby_boxes(self) -> None:
        station = DemoStation.from_path(PROJECT_ROOT / "configs" / "station_demo.json")
        tracker = AssistanceTracker()
        frame_a = [
            FrameDetection(
                class_name="wheelchair_user",
                confidence=0.8,
                bbox_xyxy=[100.0, 120.0, 180.0, 260.0],
                center_norm=(0.2, 0.4),
            )
        ]
        frame_b = [
            FrameDetection(
                class_name="wheelchair_user",
                confidence=0.82,
                bbox_xyxy=[105.0, 122.0, 185.0, 262.0],
                center_norm=(0.205, 0.405),
            )
        ]

        tracker.update(0, frame_a, station.assign_zone)
        tracker.update(1, frame_b, station.assign_zone)
        tracker.finalize()
        exported = tracker.export_tracks(min_observations=2)

        self.assertEqual(len(exported), 1)
        self.assertEqual(exported[0]["track_id"], 1)
        self.assertEqual(exported[0]["dominant_label"], "wheelchair_user")


if __name__ == "__main__":
    unittest.main()
