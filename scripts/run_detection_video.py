from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "inference"


def _json_ready_box(box, names: dict[int, str]) -> dict:
    cls_id = int(box.cls.item())
    conf = float(box.conf.item())
    x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
    return {
        "class_id": cls_id,
        "class_name": names.get(cls_id, str(cls_id)),
        "confidence": round(conf, 5),
        "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained Ultralytics detector on a video and save outputs.")
    parser.add_argument("--weights", required=True, help="Path to the trained .pt file.")
    parser.add_argument("--source", required=True, help="Path to the input video.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0", help="Use '0' for GPU or 'cpu' for CPU.")
    parser.add_argument("--name", default="video_demo")
    parser.add_argument("--project", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is not installed. Install ultralytics first.") from exc

    weights_path = Path(args.weights).resolve()
    source_path = Path(args.source).resolve()
    if not weights_path.exists():
        raise SystemExit(f"Weights file not found: {weights_path}")
    if not source_path.exists():
        raise SystemExit(f"Video source not found: {source_path}")

    output_root = Path(args.project).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    prediction_jsonl = output_root / f"{args.name}_predictions.jsonl"

    model = YOLO(str(weights_path))
    results = model.predict(
        source=str(source_path),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=True,
        save_txt=False,
        project=str(output_root),
        name=args.name,
        exist_ok=True,
        stream=True,
        verbose=False,
    )

    names = {int(key): value for key, value in model.names.items()} if isinstance(model.names, dict) else {}
    with prediction_jsonl.open("w", encoding="utf-8") as handle:
        for frame_index, result in enumerate(results):
            boxes = [_json_ready_box(box, names) for box in result.boxes]
            handle.write(
                json.dumps(
                    {
                        "frame_index": frame_index,
                        "source": str(source_path),
                        "num_detections": len(boxes),
                        "detections": boxes,
                    }
                )
                + "\n"
            )

    print(
        json.dumps(
            {
                "status": "ok",
                "weights": str(weights_path),
                "source": str(source_path),
                "annotated_output_dir": str((output_root / args.name).resolve()),
                "predictions_jsonl": str(prediction_jsonl.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
