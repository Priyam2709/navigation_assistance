from __future__ import annotations

import configparser
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from .constants import CROSSROAD_CLASS_MAPPING, MOT17_EVAL_SEQUENCES, PMMA_IGNORE_KEYWORDS
from .models import Annotation, DatasetRecord
from .utils import (
    image_size,
    infer_split,
    parse_yolo_names,
    relative_to,
    write_jsonl,
    xywh_norm_to_abs,
)


def _record_sort_key(record: DatasetRecord) -> tuple[str, str, int]:
    return (
        record.split,
        record.video_id or "",
        record.frame_index or 0,
    )


def _resolve_label_path(image_path: Path) -> Path | None:
    candidate_pairs = [
        (
            Path(str(image_path.parent).replace("\\images\\", "\\labels\\").replace("/images/", "/labels/")),
            image_path.stem + ".txt",
        ),
        (image_path.parent, image_path.stem + ".txt"),
        (image_path.parent.parent / "labels" / image_path.parent.name, image_path.stem + ".txt"),
    ]
    for directory, filename in candidate_pairs:
        label_path = directory / filename
        if label_path.exists():
            return label_path
    return None


def convert_crossroad_dataset(source_root: Path, output_path: Path, project_root: Path) -> dict[str, int]:
    if not source_root.exists():
        raise FileNotFoundError(f"Crossroad root not found: {source_root}")

    class_names = parse_yolo_names(source_root)
    image_paths = sorted(
        path for path in source_root.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    records: list[DatasetRecord] = []

    for image_path in image_paths:
        split = infer_split(image_path)
        width, height = image_size(image_path)
        label_path = _resolve_label_path(image_path)
        annotations: list[Annotation] = []
        if label_path:
            with label_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    class_idx = int(parts[0])
                    x_center, y_center, width_norm, height_norm = map(float, parts[1:5])
                    source_class = class_names[class_idx] if class_idx < len(class_names) else str(class_idx)
                    canonical_class = CROSSROAD_CLASS_MAPPING.get(source_class.strip().lower())
                    if canonical_class is None:
                        continue
                    bbox = xywh_norm_to_abs(x_center, y_center, width_norm, height_norm, width, height)
                    annotations.append(
                        Annotation(
                            class_name=canonical_class,
                            bbox_xywh_abs=bbox,
                            ignore=False,
                            source_class=source_class,
                        )
                    )

        record = DatasetRecord(
            record_id=f"crossroad::{image_path.stem}",
            source_dataset="crossroad_mobility",
            split=split,
            image_path=str(image_path.resolve()),
            image_relpath=relative_to(image_path, source_root),
            width=width,
            height=height,
            original_id=image_path.stem,
            annotations=annotations,
            metadata={"label_path": str(label_path.resolve()) if label_path else None},
        )
        records.append(record)

    records.sort(key=_record_sort_key)
    write_jsonl(output_path, [record.to_dict() for record in records])
    return {"records": len(records), "annotations": sum(len(record.annotations) for record in records)}


def convert_crowdhuman_dataset(source_root: Path, output_path: Path, project_root: Path) -> dict[str, int]:
    if not source_root.exists():
        raise FileNotFoundError(f"CrowdHuman root not found: {source_root}")

    image_index = {
        path.stem: path
        for path in source_root.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    }

    records: list[DatasetRecord] = []
    odgt_files = sorted(source_root.rglob("*.odgt"))
    for odgt_path in odgt_files:
        split = infer_split(odgt_path)
        with odgt_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                item = json.loads(raw_line)
                image_id = item["ID"]
                image_path = image_index.get(image_id)
                if image_path is None:
                    continue
                width, height = image_size(image_path)
                annotations: list[Annotation] = []
                for gtbox in item.get("gtboxes", []):
                    tag = str(gtbox.get("tag", "")).lower()
                    extra = gtbox.get("extra", {}) or {}
                    ignore = bool(extra.get("ignore", 0)) or tag != "person"
                    fbox = gtbox.get("fbox") or gtbox.get("vbox")
                    if not fbox or len(fbox) < 4:
                        continue
                    bbox = [float(fbox[0]), float(fbox[1]), float(fbox[2]), float(fbox[3])]
                    annotations.append(
                        Annotation(
                            class_name="person",
                            bbox_xywh_abs=bbox,
                            ignore=ignore,
                            source_class=tag or "person",
                            attributes={"head_attr": gtbox.get("head_attr", {})},
                        )
                    )
                records.append(
                    DatasetRecord(
                        record_id=f"crowdhuman::{image_id}",
                        source_dataset="crowdhuman",
                        split=split,
                        image_path=str(image_path.resolve()),
                        image_relpath=relative_to(image_path, source_root),
                        width=width,
                        height=height,
                        original_id=image_id,
                        annotations=annotations,
                    )
                )

    records.sort(key=_record_sort_key)
    write_jsonl(output_path, [record.to_dict() for record in records])
    return {"records": len(records), "annotations": sum(len(record.annotations) for record in records)}


def _map_pmma_class(category_name: str) -> tuple[str | None, bool]:
    label = category_name.strip().lower()
    if any(keyword in label for keyword in PMMA_IGNORE_KEYWORDS):
        return None, True
    if "wheelchair" in label:
        return "wheelchair_user", False
    if "rollator" in label or "walker" in label:
        return "walker_user", False
    if "crutch" in label:
        return "crutch_user", False
    if "cane" in label:
        return "cane_user", False
    if "pedestrian" in label or "person" in label:
        return "person", False
    return None, True


def _find_pmma_image(source_root: Path, relative_file_name: str) -> Path | None:
    normalized = relative_file_name.replace("\\", "/").lstrip("./")
    candidates = [
        source_root / normalized,
        source_root / "images" / normalized,
        source_root / Path(normalized).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tail = Path(normalized).name
    for candidate in source_root.rglob(tail):
        if candidate.is_file():
            return candidate
    return None


def convert_pmma_dataset(source_root: Path, output_path: Path, project_root: Path) -> dict[str, int]:
    if not source_root.exists():
        raise FileNotFoundError(f"PMMA root not found: {source_root}")

    annotation_files = sorted(source_root.rglob("*.json"))
    records: list[DatasetRecord] = []

    for annotation_file in annotation_files:
        split = infer_split(annotation_file)
        if split not in {"train", "val", "test"}:
            continue
        try:
            payload = json.loads(annotation_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        if not all(key in payload for key in ("images", "annotations", "categories")):
            continue

        categories = {int(item["id"]): str(item["name"]) for item in payload["categories"]}
        images = {int(item["id"]): item for item in payload["images"]}
        grouped_annotations: dict[int, list[dict]] = defaultdict(list)
        for annotation in payload["annotations"]:
            grouped_annotations[int(annotation["image_id"])].append(annotation)

        for image_id, image_meta in images.items():
            image_path = _find_pmma_image(source_root, image_meta.get("file_name", ""))
            if image_path is None or not image_path.exists():
                continue
            width = int(image_meta.get("width") or image_size(image_path)[0])
            height = int(image_meta.get("height") or image_size(image_path)[1])
            filename = str(image_meta.get("file_name", ""))
            video_id = str(image_meta.get("video_id") or Path(filename).parent.name or annotation_file.stem)
            frame_index = image_meta.get("frame_id") or image_meta.get("frame_index")

            annotations: list[Annotation] = []
            for annotation in grouped_annotations.get(image_id, []):
                category_name = categories.get(int(annotation["category_id"]), str(annotation["category_id"]))
                class_name, ignore = _map_pmma_class(category_name)
                bbox = annotation.get("bbox")
                if bbox is None or len(bbox) < 4:
                    continue
                if class_name is None:
                    annotations.append(
                        Annotation(
                            class_name="person",
                            bbox_xywh_abs=[float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                            ignore=True,
                            source_class=category_name,
                        )
                    )
                    continue
                track_id = annotation.get("track_id") or annotation.get("instance_id")
                annotations.append(
                    Annotation(
                        class_name=class_name,
                        bbox_xywh_abs=[float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                        ignore=ignore,
                        source_class=category_name,
                        track_id=str(track_id) if track_id is not None else None,
                    )
                )

            records.append(
                DatasetRecord(
                    record_id=f"pmma::{image_id}",
                    source_dataset="pmma",
                    split=split,
                    image_path=str(image_path.resolve()),
                    image_relpath=relative_to(image_path, source_root),
                    width=width,
                    height=height,
                    original_id=str(image_id),
                    video_id=video_id,
                    frame_index=int(frame_index) if frame_index is not None else None,
                    annotations=annotations,
                    metadata={"annotation_file": str(annotation_file.resolve())},
                )
            )

    records.sort(key=_record_sort_key)
    write_jsonl(output_path, [record.to_dict() for record in records])
    return {"records": len(records), "annotations": sum(len(record.annotations) for record in records)}


def convert_mot17_dataset(source_root: Path, output_path: Path, project_root: Path) -> dict[str, int]:
    if not source_root.exists():
        raise FileNotFoundError(f"MOT17 root not found: {source_root}")

    records: list[DatasetRecord] = []
    for sequence_root in sorted(path for path in source_root.iterdir() if path.is_dir()):
        sequence_name = sequence_root.name
        if sequence_name not in MOT17_EVAL_SEQUENCES:
            continue
        seqinfo_path = sequence_root / "seqinfo.ini"
        gt_path = sequence_root / "gt" / "gt.txt"
        image_dir = sequence_root / "img1"
        if not (seqinfo_path.exists() and gt_path.exists() and image_dir.exists()):
            continue

        config = configparser.ConfigParser()
        config.read(seqinfo_path, encoding="utf-8")
        section = config["Sequence"]
        width = int(section["imWidth"])
        height = int(section["imHeight"])
        ext = section.get("imExt", ".jpg")

        annotations_by_frame: dict[int, list[Annotation]] = defaultdict(list)
        with gt_path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if len(row) < 9:
                    continue
                frame_id = int(float(row[0]))
                track_id = row[1]
                left = float(row[2])
                top = float(row[3])
                box_width = float(row[4])
                box_height = float(row[5])
                confidence = float(row[6])
                class_id = int(float(row[7]))
                visibility = float(row[8])
                if class_id != 1 or confidence == 0:
                    continue
                annotations_by_frame[frame_id].append(
                    Annotation(
                        class_name="person",
                        bbox_xywh_abs=[left, top, box_width, box_height],
                        ignore=False,
                        source_class="pedestrian",
                        track_id=track_id,
                        attributes={"visibility": visibility},
                    )
                )

        for frame_id, annotations in sorted(annotations_by_frame.items()):
            image_path = image_dir / f"{frame_id:06d}{ext}"
            if not image_path.exists():
                continue
            records.append(
                DatasetRecord(
                    record_id=f"mot17::{sequence_name}::{frame_id}",
                    source_dataset="mot17",
                    split="test",
                    image_path=str(image_path.resolve()),
                    image_relpath=relative_to(image_path, source_root),
                    width=width,
                    height=height,
                    original_id=f"{sequence_name}_{frame_id:06d}",
                    video_id=sequence_name,
                    frame_index=frame_id,
                    annotations=annotations,
                )
            )

    records.sort(key=_record_sort_key)
    write_jsonl(output_path, [record.to_dict() for record in records])
    return {"records": len(records), "annotations": sum(len(record.annotations) for record in records)}


def convert_market1501_dataset(source_root: Path, output_path: Path, project_root: Path) -> dict[str, int]:
    if not source_root.exists():
        raise FileNotFoundError(f"Market1501 root not found: {source_root}")

    folders = {
        "train": source_root / "bounding_box_train",
        "query": source_root / "query",
        "gallery": source_root / "bounding_box_test",
    }
    filename_pattern = re.compile(r"([-\d]+)_c(\d)")
    rows: list[dict] = []
    for split, folder in folders.items():
        if not folder.exists():
            continue
        for image_path in sorted(folder.glob("*.jpg")):
            match = filename_pattern.search(image_path.name)
            if not match:
                continue
            person_id = match.group(1)
            camera_id = match.group(2)
            rows.append(
                {
                    "source_dataset": "market1501",
                    "split": split,
                    "image_path": str(image_path.resolve()),
                    "image_relpath": relative_to(image_path, source_root),
                    "person_id": person_id,
                    "camera_id": camera_id,
                    "original_id": image_path.stem,
                }
            )
    write_jsonl(output_path, rows)
    return {"records": len(rows), "annotations": 0}
