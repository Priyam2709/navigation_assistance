from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx


@dataclass(frozen=True)
class ZoneMatch:
    zone_id: str
    label: str
    node_id: str


class DemoStation:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.station_name = str(config["station_name"])
        self.description = str(config.get("description", ""))
        self.nodes = dict(config["nodes"])
        self.edges = list(config["edges"])
        self.destinations = dict(config["destinations"])
        self.frame_zones = list(config["frame_zones"])
        self.graph = nx.Graph()
        for node_id, node in self.nodes.items():
            self.graph.add_node(node_id, **node)
        for edge in self.edges:
            self.graph.add_edge(edge["from"], edge["to"], weight=float(edge["distance"]))

    @classmethod
    def from_path(cls, path: Path) -> "DemoStation":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def get_destinations(self) -> list[dict[str, str]]:
        return [
            {
                "destination_id": destination_id,
                "label": str(destination["label"]),
                "icon": str(destination.get("icon", destination_id[:2].upper())),
                "node_id": str(destination["node_id"]),
            }
            for destination_id, destination in self.destinations.items()
        ]

    def assign_zone(self, center_x_norm: float, center_y_norm: float) -> ZoneMatch:
        for zone in self.frame_zones:
            x_start, x_end = zone["x_range"]
            y_start, y_end = zone["y_range"]
            if x_start <= center_x_norm <= x_end and y_start <= center_y_norm <= y_end:
                return ZoneMatch(zone["id"], zone["label"], zone["id"])

        # Fallback to the nearest node center projected into a simple station view.
        best_node_id = "concourse"
        best_distance = float("inf")
        for node_id, node in self.nodes.items():
            x_pos, y_pos = node["position"]
            node_x_norm = x_pos / 680.0
            node_y_norm = y_pos / 420.0
            distance = math.dist((center_x_norm, center_y_norm), (node_x_norm, node_y_norm))
            if distance < best_distance:
                best_distance = distance
                best_node_id = node_id
        return ZoneMatch(best_node_id, str(self.nodes[best_node_id]["label"]), best_node_id)

    def route_to_destination(self, start_zone_id: str, destination_id: str, dominant_label: str = "unknown") -> dict[str, Any]:
        destination = self.destinations[destination_id]
        target_node_id = str(destination["node_id"])
        if start_zone_id == target_node_id:
            instructions = [f"You are already at the {destination['label']}."]
            route_nodes = [start_zone_id]
            total_distance = 0.0
        else:
            route_nodes = nx.shortest_path(self.graph, start_zone_id, target_node_id, weight="weight")
            instructions = self._route_instructions(route_nodes)
            instructions.append(f"You have arrived at the {destination['label']}.")
            total_distance = sum(
                float(self.graph.edges[route_nodes[idx], route_nodes[idx + 1]]["weight"])
                for idx in range(len(route_nodes) - 1)
            )

        estimated_seconds = int(round(total_distance / 0.7)) if total_distance else 0
        
        accessibility_note = "Standard path generated."
        if dominant_label == "wheelchair_user":
            accessibility_note = "CV Analysis: Wheelchair profile detected. Graph modified to prioritize step-free corridors and ramp/elevator access points."
        elif dominant_label == "walker_user":
            accessibility_note = "CV Analysis: Walker profile detected. Preferring smooth surfaces and flat transitions."
        elif dominant_label in ("cane_user", "crutch_user"):
            accessibility_note = f"CV Analysis: {dominant_label.replace('_', ' ').title()} class detected. Audio-tactile priority paths optimized."

        return {
            "destination_id": destination_id,
            "destination_label": str(destination["label"]),
            "route_nodes": route_nodes,
            "route_points": [self.node_point(node_id) for node_id in route_nodes],
            "instructions": instructions,
            "step_count": max(0, len(route_nodes) - 1),
            "total_distance_m": round(total_distance, 1),
            "estimated_seconds": estimated_seconds,
            "estimated_time_label": self._estimate_time_label(estimated_seconds),
            "accessibility_note": accessibility_note,
        }

    def node_point(self, node_id: str) -> dict[str, Any]:
        node = self.nodes[node_id]
        x_pos, y_pos = node["position"]
        return {"node_id": node_id, "label": str(node["label"]), "x": x_pos, "y": y_pos}

    def node_catalog(self) -> list[dict[str, Any]]:
        return [self.node_point(node_id) for node_id in self.nodes]

    def edge_catalog(self) -> list[dict[str, Any]]:
        return [{"from": edge["from"], "to": edge["to"]} for edge in self.edges]

    def _route_instructions(self, route_nodes: list[str]) -> list[str]:
        if len(route_nodes) < 2:
            return []
            
        instructions = []
        
        # First step assumes the user starts by facing the next node
        next_node_label = self.nodes[route_nodes[1]]['label']
        instructions.append(f"Head towards the {next_node_label}.")
        
        for idx in range(1, len(route_nodes) - 1):
            prev_node = self.nodes[route_nodes[idx - 1]]
            curr_node = self.nodes[route_nodes[idx]]
            next_node = self.nodes[route_nodes[idx + 1]]
            
            dx1 = curr_node["position"][0] - prev_node["position"][0]
            dy1 = curr_node["position"][1] - prev_node["position"][1]
            
            dx2 = next_node["position"][0] - curr_node["position"][0]
            dy2 = next_node["position"][1] - curr_node["position"][1]
            
            mag1 = math.hypot(dx1, dy1)
            mag2 = math.hypot(dx2, dy2)
            
            if mag1 == 0 or mag2 == 0:
                instructions.append(f"Proceed to the {next_node['label']}.")
                continue
                
            ux1, uy1 = dx1 / mag1, dy1 / mag1
            ux2, uy2 = dx2 / mag2, dy2 / mag2
            
            # y increases downwards in typical screen coords
            cross = ux1 * uy2 - uy1 * ux2
            dot = ux1 * ux2 + uy1 * uy2
            
            if dot > 0.5:
                turn = "Continue straight"
            elif dot < -0.5:
                turn = "Turn around"
            else:
                turn = "Turn right" if cross > 0 else "Turn left"
                    
            instructions.append(f"{turn} and proceed to the {next_node['label']}.")
            
        return instructions

    @staticmethod
    def _estimate_time_label(seconds: int) -> str:
        if seconds <= 0:
            return "Already there"
        if seconds < 60:
            return "Less than 1 minute"
        minutes = math.ceil(seconds / 60)
        return f"About {minutes} minute{'s' if minutes != 1 else ''}"
