from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import qrcode


class JsonStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.analysis_root = self.root / "analyses"
        self.session_root = self.root / "sessions"
        self.analysis_root.mkdir(parents=True, exist_ok=True)
        self.session_root.mkdir(parents=True, exist_ok=True)

    def save_analysis(self, analysis: dict) -> None:
        path = self.analysis_root / f"{analysis['analysis_id']}.json"
        path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    def load_analysis(self, analysis_id: str) -> dict | None:
        path = self.analysis_root / f"{analysis_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list_recent_analyses(self, limit: int = 8) -> list[dict]:
        items = []
        for path in sorted(self.analysis_root.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            items.append(json.loads(path.read_text(encoding="utf-8")))
            if len(items) >= limit:
                break
        return items

    def create_session(
        self,
        *,
        analysis: dict,
        track: dict,
        route: dict,
        passenger_url: str,
        qr_root: Path,
        qr_image_url: str,
        session_id: str | None = None,
    ) -> dict:
        session_id = session_id or f"session-{uuid4().hex[:10]}"
        qr_root.mkdir(parents=True, exist_ok=True)
        qr_image_path = qr_root / f"{session_id}.png"
        qrcode.make(passenger_url).save(qr_image_path)

        payload = {
            "session_id": session_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "analysis_id": analysis["analysis_id"],
            "track_id": track["track_id"],
            "dominant_label": track["dominant_label"],
            "current_zone_id": track["current_zone_id"],
            "current_zone_label": track["current_zone_label"],
            "route": route,
            "passenger_url": passenger_url,
            "qr_image_path": str(qr_image_path.resolve()),
            "qr_image_url": qr_image_url,
            "track": track,
            "station_name": analysis["station_name"],
        }
        self.save_session(payload)
        return payload

    def load_session(self, session_id: str) -> dict | None:
        path = self.session_root / f"{session_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save_session(self, session: dict) -> None:
        (self.session_root / f"{session['session_id']}.json").write_text(json.dumps(session, indent=2), encoding="utf-8")
