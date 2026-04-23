# Dataset Sources

This project uses only official public sources for the first MVP dataset pipeline.

Accessed on `2026-04-20`.

| Source | Official URL | Role In Project | Access Method | License / Terms Note | Practical Note |
| --- | --- | --- | --- | --- | --- |
| Crossroad Mobility Aid Users | [TU Graz repository](https://repository.tugraz.at/records/2gat1-pev27) | Main mobility-aid detection dataset | Public repository download | `CC BY-SA 4.0` on the official repository page | Best source for `wheelchair`, `walker`, `crutch`, `cane` user classes |
| PMMA | [Official GitHub repository](https://github.com/DatasetPMMA/PMMA) | Mobility-aid video dataset for temporal behavior and clip-based evaluation | GitHub repo plus linked dataset assets | Confirm the exact redistribution terms on the official PMMA data page before sharing derived exports | Use for mobility-aid clips and CVAT review input |
| CrowdHuman | [Official website](https://www.crowdhuman.org/) | Stronger crowded `person` detection and occlusion cases | Public site with download instructions | Official site states non-commercial research / educational usage | Do not use for mobility-aid classes |
| MOT17 | [MOTChallenge official page](https://motchallenge.net/data/MOT17/) | Tracker tuning and held-out tracking evaluation | Official benchmark download | Follow MOTChallenge dataset terms and citation requirements from the official site | Keep full sequences intact; never random-split frames |
| Market-1501 | [Official project page](https://zheng-lab-anu.github.io/Project/project_reid.html) | ReID pretraining for cross-camera identity matching | Official project page | Follow the project page terms and citation guidance; verify usage rights before redistribution | Use unchanged folder structure for ReID training |

## Why each source is separate

- `Crossroad Mobility Aid Users`: strongest direct fit to the core class map.
- `PMMA`: gives real video behavior around mobility-aid users.
- `CrowdHuman`: improves crowded-person robustness where mobility datasets are small.
- `MOT17`: provides standard tracking sequences and annotations.
- `Market-1501`: standard person ReID pretraining dataset.

## Download policy for this repo

This repository does not try to bypass gated or terms-acceptance flows.

Instead, the download helper script writes:

- a local JSON manifest with all official links
- a Markdown checklist showing where each dataset should be placed

Run:

```powershell
python scripts/download_sources.py
```

Generated files:

- `data/external/download_manifest.json`
- `data/external/DOWNLOAD_NEXT_STEPS.md`

## Required placement after download

```text
data/external/crossroad_mobility
data/external/pmma
data/external/crowdhuman
data/external/mot17
data/external/market1501
```

Keep these folders raw and untouched. All conversions must go into `data/working`.

## Crossroad direct download details

The official repository page lists:

- `mobility.zip` at `1.8 GB`
- published MD5: `2604a74426e8d8ac65c7df19d4072b47`
- `mobility_class_hierarchy.zip` at `3.9 MB`
- published MD5: `b2c0cb859cad8824c48f8c9b5f299940`

Repository page used for these details: [Crossroad Camera Dataset - Mobility Aid Users](https://repository.tugraz.at/records/2gat1-pev27)

This repo includes a helper for the official direct links:

```powershell
python scripts/download_crossroad_mobility.py --extract
```
