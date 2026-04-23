# Navigation Assistance System for Differently-Abled Passengers

This repository now contains a teacher-demoable MVP of the navigation assistance system:

- public dataset preparation
- baseline detector training
- video analysis for mobility-aid users
- tracked passenger summaries
- demo station map routing
- passenger guidance sessions with QR code access

The app is designed as a local FastAPI demo around the trained detector.

It gives you:

- a fixed dataset folder layout
- source metadata and labeling rules
- conversion scripts for the 5 public sources in the plan
- builders for detection, tracking-eval, and ReID datasets
- validation utilities for leakage checks, label-audit sheets, and loader smoke tests
- a web dashboard for video analysis
- a passenger guidance page with route instructions and browser speech

## 1. Create the Python environment

Use `Python 3.10` on this machine because it is the safest baseline for computer-vision dependencies and the current CV stack.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Then install the GPU build of PyTorch from the official selector:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## 2. Review the official source list

The source catalog and licensing notes are in [docs/dataset_sources.md](docs/dataset_sources.md).

Generate local download instructions:

```powershell
python scripts/download_sources.py
```

That command writes:

- `data/external/download_manifest.json`
- `data/external/DOWNLOAD_NEXT_STEPS.md`

Important:

- some datasets are large
- some datasets require registration or manual acceptance
- this repo does not automatically bypass gated download flows

## 3. Place the downloaded datasets in `data/external`

Expected roots:

```text
data/external/crossroad_mobility
data/external/pmma
data/external/crowdhuman
data/external/mot17
data/external/market1501
```

Do not edit the raw source folders after downloading them.

### Fastest way to fetch Crossroad on this machine

The Crossroad dataset is directly downloadable from the official TU Graz repository, so this repo includes a helper for it:

```powershell
python scripts/download_crossroad_mobility.py --extract
```

That script:

- downloads `mobility.zip` into `data/external/crossroad_mobility`
- verifies the published MD5 checksum
- extracts the archive in place

If you only want the smaller class-hierarchy labels archive:

```powershell
python scripts/download_crossroad_mobility.py --hierarchy-only --extract
```

## 4. Convert all raw datasets to the common intermediate format

```powershell
python scripts/prepare_public_dataset.py convert --source all
```

Outputs are written to `data/working/converted/*.jsonl`.

## 5. Review mobility-aid data in CVAT

Use [docs/labeling_guidelines.md](docs/labeling_guidelines.md) while reviewing the mobility datasets.

Recommended flow:

1. Import `crossroad_mobility.jsonl` and `pmma.jsonl` source images into CVAT.
2. Review labels with the canonical class policy.
3. Export your reviewed records back into JSONL files named:
   - `data/working/cvat/reviewed/crossroad_mobility.jsonl`
   - `data/working/cvat/reviewed/pmma.jsonl`

If the reviewed files exist, the build step uses them automatically.

## 6. Build the final datasets

```powershell
python scripts/prepare_public_dataset.py build
```

This creates:

- `data/final/detection`
- `data/final/tracking_eval`
- `data/final/reid_pretrain`
- split manifests in `data/working/manifests`

## 7. Run validation

```powershell
python scripts/prepare_public_dataset.py validate
```

This runs:

- split leakage checks
- duplicate hash checks
- loader smoke tests
- label audit sheet generation

Optional smoke training:

```powershell
python scripts/smoke_train_ultralytics.py --epochs 1
```

That step requires `ultralytics` and `torch`, which are intentionally not installed by default in `requirements.txt`.

## 8. Train the first detector

Then start a small training run:

```powershell
python scripts/train_detector.py --epochs 1
```

For a more useful first baseline on this laptop GPU:

```powershell
python scripts/train_detector.py --epochs 20 --batch 8
```

Outputs are written to `runs/detect/<run_name>`.

Your trained weights will be in:

```text
runs/detect/<run_name>/weights/best.pt
```

## 9. Run the detector on a video

After training, run the model on a video:

```powershell
python scripts/run_detection_video.py --weights runs\detect\crossroad_baseline\weights\best.pt --source path\to\video.mp4
```

The script saves:

- an annotated video
- frame-level prediction summaries in JSONL

Default output root:

```text
runs/inference
```

If you do not have a real video yet, create a quick demo video from test images:

```powershell
python scripts/make_demo_video.py --image-dir data\final\detection\images\test --limit 120
```

That writes a sample video under `runs/demo_video`.

## 10. Run the web app

Start the local demo server:

```powershell
python scripts/run_demo_app.py
```

Then open:

```text
http://127.0.0.1:8000
```

The fastest teacher-demo flow is:

1. Click `Analyze Bundled Demo Video`
2. Wait for the annotated video and passenger cards to appear
3. Create a guidance session for one detected passenger
4. Open the passenger page or scan the QR code
5. Use the `Speak Guidance` button to demonstrate audio-style instructions

## 10A. How to see the passenger guidance clearly

If you only want to demonstrate the final guidance output, do this:

1. Run `python scripts/run_demo_app.py`
2. Open [http://127.0.0.1:8000](http://127.0.0.1:8000)
3. Click `Analyze Bundled Demo Video`
4. On the analysis page, pick any detected passenger card
5. Select a destination and click `Create Passenger Guidance Session`
6. In the `Passenger Guidance Session` panel, click `Open Passenger Guidance Page`
7. On the passenger page, press `Speak Guidance`

What you will see on the passenger page:

- the current passenger zone
- the selected destination
- the highlighted route on the station map
- step-by-step instructions
- browser-based spoken guidance

If you want to show it on a phone, scan the QR code shown on the analysis page after the session is created.

## 10B. How to present it to a teacher

Use this 4-step explanation:

1. The model detects passengers with visible mobility aids from station video.
2. The tracker groups detections into one passenger track and estimates the passenger zone.
3. The route engine computes a path to a selected destination on the demo station map.
4. The final passenger page delivers readable and speakable guidance.

## 11. Typical beginner workflow

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python scripts/download_sources.py
python scripts/prepare_public_dataset.py convert --source crossroad_mobility
python scripts/prepare_public_dataset.py build
python scripts/prepare_public_dataset.py validate
python scripts/train_detector.py --epochs 1 --imgsz 512 --batch 4
python scripts/make_demo_video.py --image-dir data\final\detection\images\test --limit 120
python scripts/run_demo_app.py
```

## What the scripts expect

- Crossroad: YOLO-style images and labels, with class names available in a YAML file.
- PMMA: COCO-style annotation JSON plus extracted image frames.
- CrowdHuman: `annotation_train.odgt`, `annotation_val.odgt`, and matching images.
- MOT17: official sequence folders with `img1`, `gt/gt.txt`, and `seqinfo.ini`.
- Market-1501: original `bounding_box_train`, `query`, and `bounding_box_test` folders.

## Repository layout

```text
data/
  external/
  working/
  final/
docs/
scripts/
src/dataset_pipeline/
tests/
```

## What the web app does

- analyzes a short station-style video with the trained detector
- tracks mobility-aid users across frames with a lightweight internal tracker
- assigns the passenger to a demo station zone
- generates a shortest-path route to one of 5 destinations
- creates a passenger guidance session with QR-based access
- reads the instructions aloud in the browser using speech synthesis

## Notes

- This dataset pipeline is meant for academic and portfolio work.
- The current app is a local MVP and uses a demo station map plus heuristic zone assignment.
- Public data can approximate station navigation behavior, but it does not replace real railway-station footage.
- If you later commercialize the project, re-check every dataset license before reusing the prepared dataset.
