from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_YAML = PROJECT_ROOT / "data" / "final" / "detection" / "dataset.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional smoke training for the prepared YOLO dataset.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model", default="yolo11n.pt")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Install torch and ultralytics first, then rerun this script."
        ) from exc

    if not DATASET_YAML.exists():
        raise SystemExit(f"Dataset YAML not found: {DATASET_YAML}")

    model = YOLO(args.model)
    results = model.train(
        data=str(DATASET_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=str(PROJECT_ROOT / "runs"),
        name="smoke_train",
    )
    print(results)


if __name__ == "__main__":
    main()
