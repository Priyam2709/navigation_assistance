from __future__ import annotations

import csv
import random
from pathlib import Path

from PIL import Image

from .utils import file_hash, read_jsonl, write_json


def check_split_video_leakage(manifests_root: Path) -> dict[str, list[str]]:
    manifests = {
        split: read_jsonl(manifests_root / f"detection_{split}.jsonl")
        for split in ("train", "val", "test")
    }
    videos_by_split = {
        split: {row["video_id"] for row in rows if row.get("video_id")}
        for split, rows in manifests.items()
    }
    leakage: dict[str, list[str]] = {}
    splits = list(videos_by_split)
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            overlap = sorted(videos_by_split[left] & videos_by_split[right])
            if overlap:
                leakage[f"{left}_vs_{right}"] = overlap
    return leakage


def check_duplicate_hashes(manifests_root: Path, max_items_per_split: int | None = 200) -> dict[str, list[str]]:
    manifests = {
        split: read_jsonl(manifests_root / f"detection_{split}.jsonl")
        for split in ("train", "val", "test")
    }
    hashes: dict[str, list[str]] = {}
    for split, rows in manifests.items():
        subset = rows[:max_items_per_split] if max_items_per_split else rows
        for row in subset:
            image_path = Path(row["image_path"])
            if not image_path.exists():
                continue
            image_hash = file_hash(image_path)
            hashes.setdefault(image_hash, []).append(split)

    duplicates: dict[str, list[str]] = {}
    for image_hash, split_list in hashes.items():
        unique_splits = sorted(set(split_list))
        if len(unique_splits) > 1:
            duplicates[image_hash] = unique_splits
    return duplicates


def generate_label_audit_sheet(
    manifests_root: Path,
    output_csv: Path,
    sample_size: int = 200,
    seed: int = 42,
) -> dict[str, int]:
    rows = []
    for split in ("train", "val", "test"):
        rows.extend(read_jsonl(manifests_root / f"detection_{split}.jsonl"))
    if not rows:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_path", "label_path", "split", "obvious_correct", "reviewer_notes"])
        return {"sampled": 0}

    random.seed(seed)
    sample = rows if len(rows) <= sample_size else random.sample(rows, sample_size)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "label_path", "split", "obvious_correct", "reviewer_notes"])
        for row in sample:
            writer.writerow([row["image_path"], row["label_path"], row["split"], "", ""])
    return {"sampled": len(sample)}


def score_label_audit(audit_csv: Path) -> dict[str, float]:
    if not audit_csv.exists():
        return {"reviewed": 0, "pass_rate": 0.0}
    reviewed = 0
    correct = 0
    with audit_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            answer = row.get("obvious_correct", "").strip().lower()
            if not answer:
                continue
            reviewed += 1
            if answer in {"yes", "y", "1", "true"}:
                correct += 1
    pass_rate = (correct / reviewed) if reviewed else 0.0
    return {"reviewed": reviewed, "pass_rate": round(pass_rate, 4)}


def smoke_load_detection(detection_root: Path, max_items: int = 10) -> dict[str, int]:
    loaded = 0
    for split in ("train", "val", "test"):
        for image_path in sorted((detection_root / "images" / split).glob("*"))[:max_items]:
            if image_path.name == ".gitkeep":
                continue
            with Image.open(image_path) as image:
                image.verify()
            loaded += 1
    return {"loaded_images": loaded}


def smoke_load_tracking_eval(tracking_root: Path, max_items: int = 10) -> dict[str, int]:
    manifest_path = tracking_root / "tracking_eval_manifest.jsonl"
    rows = read_jsonl(manifest_path)
    loaded = 0
    for row in rows[:max_items]:
        image_path = Path(row["image_path"])
        if not image_path.exists():
            continue
        with Image.open(image_path) as image:
            image.verify()
        loaded += 1
    return {"loaded_frames": loaded}


def smoke_load_reid(reid_root: Path, max_items: int = 10) -> dict[str, int]:
    loaded = 0
    for folder_name in ("bounding_box_train", "query", "bounding_box_test"):
        for image_path in sorted((reid_root / folder_name).glob("*.jpg"))[:max_items]:
            with Image.open(image_path) as image:
                image.verify()
            loaded += 1
    return {"loaded_images": loaded}


def run_validation_suite(project_root: Path) -> dict[str, object]:
    manifests_root = project_root / "data" / "working" / "manifests"
    detection_root = project_root / "data" / "final" / "detection"
    tracking_root = project_root / "data" / "final" / "tracking_eval"
    reid_root = project_root / "data" / "final" / "reid_pretrain"

    audit_csv = manifests_root / "label_audit_sheet.csv"
    results = {
        "video_leakage": check_split_video_leakage(manifests_root),
        "duplicate_hashes": check_duplicate_hashes(manifests_root),
        "label_audit_sheet": generate_label_audit_sheet(manifests_root, audit_csv),
        "label_audit_score": score_label_audit(audit_csv),
        "detection_smoke": smoke_load_detection(detection_root),
        "tracking_smoke": smoke_load_tracking_eval(tracking_root),
        "reid_smoke": smoke_load_reid(reid_root),
    }
    write_json(manifests_root / "validation_summary.json", results)
    return results
