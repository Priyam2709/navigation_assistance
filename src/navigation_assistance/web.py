from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .analysis import analyze_video
from .settings import RUNS_ROOT, STATIC_ROOT, TEMPLATES_ROOT, load_app_config
from .station import DemoStation
from .store import JsonStore

CONFIG = load_app_config()
STATION = DemoStation.from_path(CONFIG.station_config_path)
STORE = JsonStore(CONFIG.storage_root)

app = FastAPI(title=CONFIG.app_name)
templates = Jinja2Templates(directory=str(TEMPLATES_ROOT))

STATIC_ROOT.mkdir(parents=True, exist_ok=True)
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
CONFIG.upload_root.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_ROOT)), name="static")
app.mount("/runs", StaticFiles(directory=str(RUNS_ROOT)), name="runs")


def _page_context(request: Request) -> dict:
    return {
        "request": request,
        "app_name": CONFIG.app_name,
        "station_name": STATION.station_name,
        "station_description": STATION.description,
    }


def _analysis_or_404(analysis_id: str) -> dict:
    analysis = STORE.load_analysis(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis


def _session_or_404(session_id: str) -> dict:
    session = STORE.load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return _hydrate_session(session)


def _hydrate_session(session: dict | None) -> dict | None:
    if not session:
        return None

    route = dict(session.get("route", {}))
    needs_refresh = any(
        key not in route
        for key in ("step_count", "total_distance_m", "estimated_seconds", "estimated_time_label", "accessibility_note")
    )
    destination_id = route.get("destination_id")
    current_zone_id = session.get("current_zone_id")
    dominant_label = session.get("dominant_label", "unknown")
    if needs_refresh and destination_id and current_zone_id:
        refreshed_route = STATION.route_to_destination(str(current_zone_id), str(destination_id), str(dominant_label))
        session = dict(session)
        session["route"] = refreshed_route
    return session


@app.get("/")
def dashboard(request: Request):
    recent_analyses = STORE.list_recent_analyses()
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            **_page_context(request),
            "recent_analyses": recent_analyses,
            "demo_ready": CONFIG.demo_video_path.exists() and CONFIG.weights_path.exists(),
            "weights_path": str(CONFIG.weights_path),
            "demo_video_path": str(CONFIG.demo_video_path),
            "destinations": STATION.get_destinations(),
            "station_nodes": STATION.node_catalog(),
            "station_edges": STATION.edge_catalog(),
            "analysis_count": len(recent_analyses),
        },
    )


@app.post("/analyze/demo")
def analyze_demo():
    if not CONFIG.demo_video_path.exists():
        raise HTTPException(status_code=404, detail="Demo video not found")
    analysis = analyze_video(
        project_root=Path(__file__).resolve().parents[2],
        source_path=CONFIG.demo_video_path,
        output_root=CONFIG.storage_root / "artifacts",
        station=STATION,
        weights_path=CONFIG.weights_path,
        conf=CONFIG.inference.conf,
        imgsz=CONFIG.inference.imgsz,
        device=CONFIG.inference.device,
        max_frames=CONFIG.inference.max_frames,
        min_track_observations=CONFIG.inference.min_track_observations,
    )
    STORE.save_analysis(analysis)
    return RedirectResponse(url=f"/analysis/{analysis['analysis_id']}", status_code=303)


@app.post("/analyze/upload")
async def analyze_upload(video_file: UploadFile = File(...)):
    upload_suffix = Path(video_file.filename or "upload.mp4").suffix or ".mp4"
    upload_name = f"upload-{uuid4().hex[:10]}{upload_suffix}"
    saved_path = CONFIG.upload_root / upload_name
    with saved_path.open("wb") as handle:
        shutil.copyfileobj(video_file.file, handle)

    analysis = analyze_video(
        project_root=Path(__file__).resolve().parents[2],
        source_path=saved_path,
        output_root=CONFIG.storage_root / "artifacts",
        station=STATION,
        weights_path=CONFIG.weights_path,
        conf=CONFIG.inference.conf,
        imgsz=CONFIG.inference.imgsz,
        device=CONFIG.inference.device,
        max_frames=CONFIG.inference.max_frames,
        min_track_observations=CONFIG.inference.min_track_observations,
    )
    STORE.save_analysis(analysis)
    return RedirectResponse(url=f"/analysis/{analysis['analysis_id']}", status_code=303)


@app.get("/analysis/{analysis_id}")
def analysis_page(request: Request, analysis_id: str, session_id: str | None = None):
    analysis = _analysis_or_404(analysis_id)
    selected_session = _hydrate_session(STORE.load_session(session_id)) if session_id else None
    return templates.TemplateResponse(
        request=request,
        name="analysis.html",
        context={
            **_page_context(request),
            "analysis": analysis,
            "station_nodes": STATION.node_catalog(),
            "station_edges": STATION.edge_catalog(),
            "destinations": STATION.get_destinations(),
            "selected_session": selected_session,
        },
    )


@app.post("/analysis/{analysis_id}/session")
def create_session(
    request: Request,
    analysis_id: str,
    track_id: int = Form(...),
    destination_id: str = Form(...),
):
    analysis = _analysis_or_404(analysis_id)
    track = next((item for item in analysis["passengers"] if int(item["track_id"]) == int(track_id)), None)
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    if destination_id not in STATION.destinations:
        raise HTTPException(status_code=400, detail="Destination not found")

    route = STATION.route_to_destination(track["current_zone_id"], destination_id, track.get("dominant_label", "unknown"))
    session_hint = f"session-{uuid4().hex[:10]}"
    passenger_url = str(request.base_url).rstrip("/") + f"/passenger/{session_hint}"
    qr_image_url = f"/runs/app_state/qr_codes/{session_hint}.png"
    session = STORE.create_session(
        analysis=analysis,
        track=track,
        route=route,
        passenger_url=passenger_url,
        qr_root=CONFIG.storage_root / "qr_codes",
        qr_image_url=qr_image_url,
        session_id=session_hint,
    )
    return RedirectResponse(url=f"/analysis/{analysis_id}?session_id={session['session_id']}", status_code=303)


@app.get("/passenger/{session_id}")
def passenger_page(request: Request, session_id: str):
    session = _session_or_404(session_id)
    return templates.TemplateResponse(
        request=request,
        name="passenger.html",
        context={
            **_page_context(request),
            "session": session,
            "station_nodes": STATION.node_catalog(),
            "station_edges": STATION.edge_catalog(),
        },
    )


@app.get("/api/session/{session_id}")
def session_api(session_id: str):
    return _session_or_404(session_id)


@app.get("/api/analysis/{analysis_id}")
def analysis_api(analysis_id: str):
    return _analysis_or_404(analysis_id)
