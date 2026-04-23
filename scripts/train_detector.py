from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_YAML = PROJECT_ROOT / "data" / "final" / "detection" / "dataset.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the first object detector on the prepared dataset.")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model checkpoint to fine-tune.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="Use '0' for the first GPU or 'cpu' for CPU.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="crossroad_baseline")
    parser.add_argument("--project", default=str(PROJECT_ROOT / "runs" / "detect"))
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--cache", action="store_true", help="Cache images for faster repeated training.")
    args = parser.parse_args()

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependencies. Install torch, torchvision, and ultralytics first."
        ) from exc

    if not DATASET_YAML.exists():
        raise SystemExit(f"Dataset YAML not found: {DATASET_YAML}")

    print(
        json.dumps(
            {
                "dataset": str(DATASET_YAML),
                "cuda_available": torch.cuda.is_available(),
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "project": args.project,
                "run_name": args.name,
            },
            indent=2,
        )
    )

    model = YOLO(args.model)
    results = model.train(
        data=str(DATASET_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
        exist_ok=True,
        cache=args.cache,
    )
    print(results)


if __name__ == "__main__":
    main()
