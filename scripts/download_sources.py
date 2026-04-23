from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dataset_pipeline.constants import SOURCE_CATALOG, SOURCE_DATE
from dataset_pipeline.utils import ensure_dir, write_json


def build_manifest() -> dict:
    return {
        "generated_on": SOURCE_DATE,
        "project_root": str(PROJECT_ROOT.resolve()),
        "sources": [
            {
                "name": source.name,
                "folder": source.folder,
                "official_url": source.official_url,
                "role": source.role,
                "access": source.access,
                "license_note": source.license_note,
                "notes": source.notes,
                "destination": str((PROJECT_ROOT / "data" / "external" / source.folder).resolve()),
            }
            for source in SOURCE_CATALOG
        ],
    }


def build_markdown(manifest: dict) -> str:
    lines = [
        "# Download Next Steps",
        "",
        "Use only the official URLs listed below.",
        "",
        "After downloading each source, place it in the destination folder shown for that source.",
        "",
    ]
    for source in manifest["sources"]:
        lines.extend(
            [
                f"## {source['name']}",
                "",
                f"- Official URL: {source['official_url']}",
                f"- Destination folder: `{source['destination']}`",
                f"- Role: {source['role']}",
                f"- Access method: {source['access']}",
                f"- License / terms note: {source['license_note']}",
                f"- Practical note: {source['notes']}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    external_root = ensure_dir(PROJECT_ROOT / "data" / "external")
    manifest = build_manifest()
    write_json(external_root / "download_manifest.json", manifest)
    (external_root / "DOWNLOAD_NEXT_STEPS.md").write_text(build_markdown(manifest), encoding="utf-8")
    print(json.dumps({"status": "ok", "manifest": str((external_root / 'download_manifest.json').resolve())}, indent=2))


if __name__ == "__main__":
    main()
