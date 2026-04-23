from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from navigation_assistance.station import DemoStation


class StationRoutingTests(unittest.TestCase):
    def test_station_route_to_platform(self) -> None:
        station = DemoStation.from_path(PROJECT_ROOT / "configs" / "station_demo.json")
        route = station.route_to_destination("ticket_counter", "platform_1")
        self.assertEqual(route["route_nodes"][0], "ticket_counter")
        self.assertEqual(route["route_nodes"][-1], "platform_1")
        self.assertTrue(route["instructions"])

    def test_zone_assignment(self) -> None:
        station = DemoStation.from_path(PROJECT_ROOT / "configs" / "station_demo.json")
        zone = station.assign_zone(0.1, 0.2)
        self.assertEqual(zone.zone_id, "ticket_counter")


if __name__ == "__main__":
    unittest.main()
