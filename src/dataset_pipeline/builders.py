from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from .constants import CANONICAL_CLASSES, CLASS_TO_ID
from .models import DatasetRecord
from .utils import (
    ensure_dir,
    image_size,
    link_or_copy,
    read_jsonl,
    resolve_reviewed_manifest,
    safe_name,
    write_json,
    write_jsonl,
)


def write_detection_metadata(detection_root: Path) -> None:
    ensure_dir(detection_root)
    detection_root_abs = str(detection_root.resolve()).replace("\\", "/")
    dataset_yaml = (
        f"path: {detection_root_abs}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        f"  0: {CANONICAL_CLASSES[0]}\n"
        f"  1: {CANONICAL_CLASSES[1]}\n"
        f"  2: {CANONICAL_CLASSES[2]}\n"
        f"  3: {CANONICAL_CLASSES[3]}\n"
        f"  4: {CANONICAL_CLASSES[4]}\n"
    )
    (detection_root / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")
    write_json(
        detection_root / "class_map.json",
        {
            "classes": CANONICAL_CLASSES,
            "class_to_id": CLASS_TO_ID,
            "note": "The v1 dataset is user-centric, not aid-object-centric.",
        },
    )


def bootstrap_workspace(project_root: Path) -> None:
    keepers = [
        project_root / "data" / "external" / "crossroad_mobility" / ".gitkeep",
        project_root / "data" / "external" / "pmma" / ".gitkeep",
        project_root / "data" / "external" / "crowdhuman" / ".gitkeep",
        project_root / "data" / "external" / "mot17" / ".gitkeep",
        project_root / "data" / "external" / "market1501" / ".gitkeep",
        project_root / "data" / "working" / "cvat" / ".gitkeep",
        project_root / "data" / "working" / "converted" / ".gitkeep",
        project_root / "data" / "working" / "manifests" / ".gitkeep",
        project_root / "data" / "final" / "detection" / "images" / "train" / ".gitkeep",
        project_root / "data" / "final" / "detection" / "images" / "val" / ".gitkeep",
        project_root / "data" / "final" / "detection" / "images" / "test" / ".gitkeep",
        project_root / "data" / "final" / "detection" / "labels" / "train" / ".gitkeep",
        project_root / "data" / "final" / "detection" / "labels" / "val" / ".gitkeep",
        project_root / "data" / "final" / "detection" / "labels" / "test" / ".gitkeep",
        project_root / "data" / "final" / "tracking_eval" / ".gitkeep",
        project_root / "data" / "final" / "reid_pretrain" / ".gitkeep",
    ]
    for keeper in keepers:
        ensure_dir(keeper.parent)
        keeper.write_text("", encoding="utf-8")

    write_detection_metadata(project_root / "data" / "final" / "detection")


def _load_records(path: Path) -> list[DatasetRecord]:
    return [DatasetRecord.from_dict(item) for item in read_jsonl(path)]


def _load_detection_source_records(project_root: Path, prefer_reviewed: bool = True) -> list[DatasetRecord]:
    converted_root = project_root / "data" / "working" / "converted"
    source_names = ["crossroad_mobility", "pmma", "crowdhuman"]
    records: list[DatasetRecord] = []
    for source_name in source_names:
        reviewed_path = resolve_reviewed_manifest(project_root, source_name)
        source_path = reviewed_path if prefer_reviewed and reviewed_path.exists() else converted_root / f"{source_name}.jsonl"
        if source_path.exists():
            records.extend(_load_records(source_path))
    return records


def _yolo_line(class_name: str, bbox_xywh_abs: list[float], width: int, height: int) -> str:
    left, top, box_width, box_height = bbox_xywh_abs
    x_center = (left + (box_width / 2.0)) / width
    y_center = (top + (box_height / 2.0)) / height
    return f"{CLASS_TO_ID[class_name]} {x_center:.6f} {y_center:.6f} {box_width / width:.6f} {box_height / height:.6f}"


def build_detection_dataset(project_root: Path, link_mode: str = "hardlink", prefer_reviewed: bool = True) -> dict[str, dict]:
    detection_root = project_root / "data" / "final" / "detection"
    manifests_root = project_root / "data" / "working" / "manifests"
    write_detection_metadata(detection_root)

    for split in ("train", "val", "test"):
        ensure_dir(detection_root / "images" / split)
        ensure_dir(detection_root / "labels" / split)

    summary = {
        split: {"images": 0, "annotations": 0, "ignored_annotations": 0, "classes": Counter(), "skipped": 0}
        for split in ("train", "val", "test")
    }
    split_manifest_rows: dict[str, list[dict]] = defaultdict(list)
    used_names: set[str] = set()

    for record in _load_detection_source_records(project_root, prefer_reviewed=prefer_reviewed):
        image_path = Path(record.image_path)
        if not image_path.exists():
            summary[record.split]["skipped"] += 1
            continue
        width = int(record.width) if record.width else image_size(image_path)[0]
        height = int(record.height) if record.height else image_size(image_path)[1]
        split = record.split if record.split in {"train", "val", "test"} else "train"
        output_base = safe_name(f"{record.source_dataset}__{record.original_id}")
        while output_base in used_names:
            output_base = f"{output_base}_dup"
        used_names.add(output_base)
        image_destination = detection_root / "images" / split / f"{output_base}{image_path.suffix.lower()}"
        label_destination = detection_root / "labels" / split / f"{output_base}.txt"

        link_or_copy(image_path, image_destination, mode=link_mode)

        label_lines: list[str] = []
        for annotation in record.annotations:
            if annotation.ignore:
                summary[split]["ignored_annotations"] += 1
                continue
            if annotation.class_name not in CLASS_TO_ID:
                continue
            label_lines.append(_yolo_line(annotation.class_name, annotation.bbox_xywh_abs, width, height))
            summary[split]["annotations"] += 1
            summary[split]["classes"][annotation.class_name] += 1
        label_destination.write_text("\n".join(label_lines), encoding="utf-8")

        summary[split]["images"] += 1
        split_manifest_rows[split].append(
            {
                "record_id": record.record_id,
                "source_dataset": record.source_dataset,
                "original_id": record.original_id,
                "image_path": str(image_destination.resolve()),
                "label_path": str(label_destination.resolve()),
                "split": split,
                "video_id": record.video_id,
                "frame_index": record.frame_index,
            }
        )

    for split in ("train", "val", "test"):
        write_jsonl(manifests_root / f"detection_{split}.jsonl", split_manifest_rows[split])

    json_ready_summary = {
        split: {
            **values,
            "classes": dict(values["classes"]),
        }
        for split, values in summary.items()
    }
    write_json(detection_root / "dataset_report.json", json_ready_summary)
    return json_ready_summary


def build_tracking_eval_dataset(project_root: Path) -> dict[str, int]:
    converted_root = project_root / "data" / "working" / "converted"
    manifests_root = project_root / "data" / "working" / "manifests"
    tracking_root = project_root / "data" / "final" / "tracking_eval"
    ensure_dir(tracking_root)

    pmma_path = converted_root / "pmma.jsonl"
    mot17_path = converted_root / "mot17.jsonl"
    pmma_records = [record for record in _load_records(pmma_path) if record.split in {"val", "test"}] if pmma_path.exists() else []
    mot17_records = _load_records(mot17_path) if mot17_path.exists() else []

    combined_rows = []
    for record in pmma_records + mot17_records:
        combined_rows.append(
            {
                "record_id": record.record_id,
                "source_dataset": record.source_dataset,
                "split": record.split,
                "image_path": record.image_path,
                "video_id": record.video_id,
                "frame_index": record.frame_index,
                "num_annotations": len(record.annotations),
            }
        )

    write_jsonl(tracking_root / "tracking_eval_manifest.jsonl", combined_rows)
    write_jsonl(manifests_root / "tracking_eval.jsonl", combined_rows)
    return {"records": len(combined_rows), "pmma_records": len(pmma_records), "mot17_records": len(mot17_records)}


def build_reid_dataset(project_root: Path, link_mode: str = "hardlink") -> dict[str, int]:
    source_root = project_root / "data" / "external" / "market1501"
    output_root = project_root / "data" / "final" / "reid_pretrain"
    manifests_root = project_root / "data" / "working" / "manifests"
    ensure_dir(output_root)

    split_folders = {
        "train": "bounding_box_train",
        "query": "query",
        "gallery": "bounding_box_test",
    }
    copied = 0
    metadata_rows = []
    for split, folder_name in split_folders.items():
        source_folder = source_root / folder_name
        destination_folder = output_root / folder_name
        ensure_dir(destination_folder)
        if not source_folder.exists():
            continue
        for image_path in sorted(source_folder.glob("*.jpg")):
            destination = destination_folder / image_path.name
            link_or_copy(image_path, destination, mode=link_mode)
            copied += 1
            metadata_rows.append(
                {
                    "split": split,
                    "image_path": str(destination.resolve()),
                    "original_image_path": str(image_path.resolve()),
                    "filename": image_path.name,
                }
            )

    write_jsonl(output_root / "metadata.jsonl", metadata_rows)
    write_jsonl(manifests_root / "reid_pretrain.jsonl", metadata_rows)
    return {"images": copied}
