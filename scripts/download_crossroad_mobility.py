from __future__ import annotations

import argparse
import hashlib
import sys
import time
import zipfile
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = PROJECT_ROOT / "data" / "external" / "crossroad_mobility"

MOBILITY_URL = "https://repository.tugraz.at/records/2gat1-pev27/files/mobility.zip?download=1"
MOBILITY_MD5 = "2604a74426e8d8ac65c7df19d4072b47"
HIERARCHY_URL = "https://repository.tugraz.at/records/2gat1-pev27/files/mobility_class_hierarchy.zip?download=1"
HIERARCHY_MD5 = "b2c0cb859cad8824c48f8c9b5f299940"


def md5sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", "0"))
        downloaded = 0
        last_report_time = 0.0
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                should_report = downloaded == total or (now - last_report_time) >= 1.5
                if should_report:
                    if total:
                        percent = (downloaded / total) * 100
                        print(
                            f"\rDownloading {destination.name}: {percent:6.2f}% "
                            f"({downloaded // (1024 * 1024)} MB / {total // (1024 * 1024)} MB)",
                            end="",
                            flush=True,
                        )
                    else:
                        print(
                            f"\rDownloading {destination.name}: {downloaded // (1024 * 1024)} MB",
                            end="",
                            flush=True,
                        )
                    last_report_time = now
    print()


def extract_zip(archive_path: Path, destination_dir: Path) -> None:
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(destination_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the official Crossroad Mobility Aid Users dataset.")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="Destination folder for the dataset.")
    parser.add_argument("--extract", action="store_true", help="Extract the downloaded zip after verification.")
    parser.add_argument("--hierarchy-only", action="store_true", help="Download only the small class-hierarchy archive.")
    parser.add_argument("--skip-md5", action="store_true", help="Skip checksum verification.")
    args = parser.parse_args()

    url = HIERARCHY_URL if args.hierarchy_only else MOBILITY_URL
    expected_md5 = HIERARCHY_MD5 if args.hierarchy_only else MOBILITY_MD5
    filename = "mobility_class_hierarchy.zip" if args.hierarchy_only else "mobility.zip"
    destination = args.dest / filename

    print(f"Official source: {url}")
    print(f"Saving to: {destination}")
    download_file(url, destination)

    if not args.skip_md5:
        actual_md5 = md5sum(destination)
        print(f"Expected MD5: {expected_md5}")
        print(f"Actual MD5:   {actual_md5}")
        if actual_md5 != expected_md5:
            raise SystemExit("MD5 mismatch. Delete the zip and retry the download.")

    if args.extract:
        extract_dir = args.dest / filename.removesuffix(".zip")
        extract_dir.mkdir(parents=True, exist_ok=True)
        extract_zip(destination, extract_dir)
        print(f"Extracted to: {extract_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("Download interrupted by user.")
