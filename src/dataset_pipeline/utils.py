from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import yaml
from PIL import Image

SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def infer_split(path_like: str | Path, default: str = "train") -> str:
    text = str(path_like).replace("\\", "/").lower()
    for piece in re.split(r"[/_.-]+", text):
        mapped = SPLIT_ALIASES.get(piece)
        if mapped:
            return mapped
    return default


def safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "item"


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def file_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_yolo_names(root: Path) -> list[str]:
    candidates = list(root.rglob("*.yaml")) + list(root.rglob("*.yml"))
    for candidate in candidates:
        try:
            payload = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        names = payload.get("names")
        if isinstance(names, dict):
            ordered = [names[index] for index in sorted(names)]
            return [str(item) for item in ordered]
        if isinstance(names, list):
            return [str(item) for item in names]
    return []


def xywh_norm_to_abs(
    x_center: float,
    y_center: float,
    width_norm: float,
    height_norm: float,
    width_px: int,
    height_px: int,
) -> list[float]:
    width = width_norm * width_px
    height = height_norm * height_px
    left = (x_center * width_px) - (width / 2.0)
    top = (y_center * height_px) - (height / 2.0)
    return [round(left, 3), round(top, 3), round(width, 3), round(height, 3)]


def xyxy_to_xywh(left: float, top: float, right: float, bottom: float) -> list[float]:
    return [left, top, right - left, bottom - top]


def link_or_copy(source: Path, destination: Path, mode: str = "hardlink") -> None:
    ensure_dir(destination.parent)
    if destination.exists():
        return
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def resolve_reviewed_manifest(project_root: Path, source_name: str) -> Path:
    return project_root / "data" / "working" / "cvat" / "reviewed" / f"{source_name}.jsonl"


def relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def placeholder_keep_file(path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text("", encoding="utf-8")
