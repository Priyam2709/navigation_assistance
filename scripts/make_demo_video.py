from __future__ import annotations

import argparse
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data" / "final" / "detection" / "images" / "test"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "demo_video"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a simple MP4 demo video from a folder of images.")
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_ROOT / "crossroad_demo.mp4"))
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--limit", type=int, default=120)
    args = parser.parse_args()

    image_dir = Path(args.image_dir).resolve()
    output_path = Path(args.output).resolve()
    if not image_dir.exists():
        raise SystemExit(f"Image directory not found: {image_dir}")

    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not image_paths:
        raise SystemExit(f"No images found in: {image_dir}")

    selected = image_paths[: args.limit] if args.limit > 0 else image_paths
    first_frame = cv2.imread(str(selected[0]))
    if first_frame is None:
        raise SystemExit(f"Could not read first image: {selected[0]}")

    height, width = first_frame.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )

    written = 0
    for image_path in selected:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)
        written += 1

    writer.release()
    print(
        {
            "status": "ok",
            "frames_written": written,
            "fps": args.fps,
            "output_video": str(output_path),
        }
    )


if __name__ == "__main__":
    main()
