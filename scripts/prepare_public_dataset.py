from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dataset_pipeline.builders import (
    bootstrap_workspace,
    build_detection_dataset,
    build_reid_dataset,
    build_tracking_eval_dataset,
)
from dataset_pipeline.converters import (
    convert_crossroad_dataset,
    convert_crowdhuman_dataset,
    convert_market1501_dataset,
    convert_mot17_dataset,
    convert_pmma_dataset,
)
from dataset_pipeline.validators import run_validation_suite


def convert_sources(selected_source: str) -> dict[str, dict[str, int]]:
    external_root = PROJECT_ROOT / "data" / "external"
    converted_root = PROJECT_ROOT / "data" / "working" / "converted"
    converted_root.mkdir(parents=True, exist_ok=True)

    source_map = {
        "crossroad_mobility": (
            convert_crossroad_dataset,
            external_root / "crossroad_mobility",
            converted_root / "crossroad_mobility.jsonl",
        ),
        "pmma": (
            convert_pmma_dataset,
            external_root / "pmma",
            converted_root / "pmma.jsonl",
        ),
        "crowdhuman": (
            convert_crowdhuman_dataset,
            external_root / "crowdhuman",
            converted_root / "crowdhuman.jsonl",
        ),
        "mot17": (
            convert_mot17_dataset,
            external_root / "mot17",
            converted_root / "mot17.jsonl",
        ),
        "market1501": (
            convert_market1501_dataset,
            external_root / "market1501",
            converted_root / "market1501.jsonl",
        ),
    }

    names = list(source_map) if selected_source == "all" else [selected_source]
    results: dict[str, dict[str, int]] = {}
    for name in names:
        converter, source_root, output_path = source_map[name]
        results[name] = converter(source_root, output_path, PROJECT_ROOT)
    return results


def build_all(link_mode: str, prefer_reviewed: bool) -> dict[str, object]:
    detection_summary = build_detection_dataset(PROJECT_ROOT, link_mode=link_mode, prefer_reviewed=prefer_reviewed)
    tracking_summary = build_tracking_eval_dataset(PROJECT_ROOT)
    reid_summary = build_reid_dataset(PROJECT_ROOT, link_mode=link_mode)
    return {
        "detection": detection_summary,
        "tracking_eval": tracking_summary,
        "reid_pretrain": reid_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the public-data-only dataset pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("bootstrap", help="Create keep files and write dataset metadata.")

    convert_parser = subparsers.add_parser("convert", help="Convert raw datasets to the common intermediate JSONL format.")
    convert_parser.add_argument(
        "--source",
        choices=["all", "crossroad_mobility", "pmma", "crowdhuman", "mot17", "market1501"],
        default="all",
    )

    build_parser = subparsers.add_parser("build", help="Build detection, tracking-eval, and reid-pretrain outputs.")
    build_parser.add_argument("--link-mode", choices=["hardlink", "copy"], default="hardlink")
    build_parser.add_argument("--prefer-reviewed", action="store_true", default=True)
    build_parser.add_argument("--no-prefer-reviewed", dest="prefer_reviewed", action="store_false")

    validate_parser = subparsers.add_parser("validate", help="Run leakage checks and loader smoke tests.")
    validate_parser.add_argument("--output-json", default=None)

    all_parser = subparsers.add_parser("all", help="Run bootstrap, convert, build, and validate in order.")
    all_parser.add_argument("--link-mode", choices=["hardlink", "copy"], default="hardlink")
    all_parser.add_argument("--prefer-reviewed", action="store_true", default=True)
    all_parser.add_argument("--no-prefer-reviewed", dest="prefer_reviewed", action="store_false")

    args = parser.parse_args()

    if args.command == "bootstrap":
        bootstrap_workspace(PROJECT_ROOT)
        result = {"status": "ok", "step": "bootstrap"}
    elif args.command == "convert":
        result = convert_sources(args.source)
    elif args.command == "build":
        result = build_all(args.link_mode, args.prefer_reviewed)
    elif args.command == "validate":
        result = run_validation_suite(PROJECT_ROOT)
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    elif args.command == "all":
        bootstrap_workspace(PROJECT_ROOT)
        convert_sources("all")
        build_all(args.link_mode, args.prefer_reviewed)
        result = run_validation_suite(PROJECT_ROOT)
    else:
        raise ValueError(f"Unsupported command: {args.command}")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
