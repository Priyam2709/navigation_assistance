# Navigation Assistance System for Differently-Abled Passengers

This repository contains a teacher-demoable MVP of the navigation assistance system. It includes a pre-trained baseline detector, a bundled demo video, and a local FastAPI demo web application.

## 🚀 Quick Start: Running the Demo App

If you just want to run the application to see the demo, follow these simple steps.

### Prerequisites
- Install [Python 3.10+](https://www.python.org/downloads/) (Make sure to check "Add Python to PATH" during installation if you are on Windows).
- (Optional but Recommended) Git installed on your device.

### Step 1: Download the Project
Clone this repository to your device:
```bash
git clone https://github.com/Priyam2709/navigation_assistance.git
cd navigation_assistance
```
*(If you don't have Git, you can click the green "Code" button on GitHub and select "Download ZIP", then extract it and open a terminal inside the extracted folder).*

### Step 2: Set up a Virtual Environment
It is highly recommended to use a virtual environment to avoid conflicts with other Python projects.

**On Windows:**
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```
*(Note: If you get an error about execution policies on Windows, run `Set-ExecutionPolicy Unrestricted -Scope CurrentUser` in PowerShell as Administrator, or just use `.\.venv\Scripts\python` directly instead of activating).*

**On macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
Install all required libraries for the web app and object detection logic:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

*(Note: The above command will install a CPU-compatible version of PyTorch which is sufficient for running the demo on most modern laptops. If you have a dedicated NVIDIA GPU and want faster inference, you can install the CUDA version of PyTorch by running: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`)*

### Step 4: Run the Application
Start the local demo server:
```bash
python scripts/run_demo_app.py
```

### Step 5: View the App
Open your web browser and navigate to:
**http://127.0.0.1:8005/**

**How to test the Demo:**
1. Click **"Analyze Bundled Demo Video"**.
2. Wait for the annotated video and passenger cards to appear.
3. Choose a destination and create a guidance session for one detected passenger.
4. Open the passenger page (or scan the QR code).
5. Click **"Speak Guidance"** to hear the audio instructions.

---

## 🛠 Advanced Usage & Development

The rest of this guide is for researchers or developers who want to download the original datasets, run the conversion pipelines, and train the YOLO detection model from scratch.

### 1. Create the Python environment

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
http://127.0.0.1:8005
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
2. Open [http://127.0.0.1:8005](http://127.0.0.1:8005)
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
