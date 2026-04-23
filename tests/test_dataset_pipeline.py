from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dataset_pipeline.builders import build_detection_dataset, bootstrap_workspace
from dataset_pipeline.converters import (
    convert_crossroad_dataset,
    convert_crowdhuman_dataset,
    convert_market1501_dataset,
)
from dataset_pipeline.utils import read_jsonl


def make_image(path: Path, size: tuple[int, int] = (100, 100)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(120, 120, 120)).save(path)


class DatasetPipelineTests(unittest.TestCase):
    def test_convert_crossroad_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project_root = root / "project"
            dataset_root = project_root / "data" / "external" / "crossroad_mobility"
            image_path = dataset_root / "images" / "train" / "sample.jpg"
            label_path = dataset_root / "labels" / "train" / "sample.txt"
            yaml_path = dataset_root / "dataset.yaml"
            make_image(image_path)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("1 0.5 0.5 0.4 0.4\n", encoding="utf-8")
            yaml_path.write_text("names: ['pedestrian', 'wheelchair']\n", encoding="utf-8")

            output_path = project_root / "data" / "working" / "converted" / "crossroad_mobility.jsonl"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            summary = convert_crossroad_dataset(dataset_root, output_path, project_root)

            rows = read_jsonl(output_path)
            self.assertEqual(summary["records"], 1)
            self.assertEqual(rows[0]["annotations"][0]["class_name"], "wheelchair_user")

    def test_convert_crowdhuman_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project_root = root / "project"
            dataset_root = project_root / "data" / "external" / "crowdhuman"
            image_path = dataset_root / "Images" / "crowd_001.jpg"
            make_image(image_path, size=(200, 100))
            odgt = {
                "ID": "crowd_001",
                "gtboxes": [{"fbox": [10, 20, 30, 40], "tag": "person", "extra": {"ignore": 0}}],
            }
            odgt_path = dataset_root / "annotation_train.odgt"
            odgt_path.parent.mkdir(parents=True, exist_ok=True)
            odgt_path.write_text(json.dumps(odgt) + "\n", encoding="utf-8")

            output_path = project_root / "data" / "working" / "converted" / "crowdhuman.jsonl"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            summary = convert_crowdhuman_dataset(dataset_root, output_path, project_root)

            rows = read_jsonl(output_path)
            self.assertEqual(summary["records"], 1)
            self.assertEqual(rows[0]["annotations"][0]["class_name"], "person")

    def test_convert_market1501_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project_root = root / "project"
            dataset_root = project_root / "data" / "external" / "market1501"
            make_image(dataset_root / "bounding_box_train" / "0001_c1s1_001051_00.jpg")
            output_path = project_root / "data" / "working" / "converted" / "market1501.jsonl"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            summary = convert_market1501_dataset(dataset_root, output_path, project_root)
            rows = read_jsonl(output_path)

            self.assertEqual(summary["records"], 1)
            self.assertEqual(rows[0]["person_id"], "0001")

    def test_build_detection_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            bootstrap_workspace(project_root)
            image_path = project_root / "source.jpg"
            make_image(image_path, size=(100, 100))
            converted_path = project_root / "data" / "working" / "converted" / "crossroad_mobility.jsonl"
            converted_path.parent.mkdir(parents=True, exist_ok=True)
            converted_path.write_text(
                json.dumps(
                    {
                        "record_id": "crossroad::sample",
                        "source_dataset": "crossroad_mobility",
                        "split": "train",
                        "image_path": str(image_path.resolve()),
                        "image_relpath": "sample.jpg",
                        "width": 100,
                        "height": 100,
                        "original_id": "sample",
                        "annotations": [
                            {
                                "class_name": "wheelchair_user",
                                "bbox_xywh_abs": [10, 10, 40, 40],
                                "ignore": False,
                                "source_class": "wheelchair",
                                "track_id": None,
                                "attributes": {},
                            }
                        ],
                        "metadata": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = build_detection_dataset(project_root)
            image_outputs = list((project_root / "data" / "final" / "detection" / "images" / "train").glob("*.jpg"))
            label_outputs = list((project_root / "data" / "final" / "detection" / "labels" / "train").glob("*.txt"))

            self.assertEqual(summary["train"]["images"], 1)
            self.assertEqual(len(image_outputs), 1)
            self.assertEqual(len(label_outputs), 1)


if __name__ == "__main__":
    unittest.main()
