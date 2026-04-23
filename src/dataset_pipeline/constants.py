from __future__ import annotations

from dataclasses import dataclass

CANONICAL_CLASSES = [
    "person",
    "wheelchair_user",
    "walker_user",
    "crutch_user",
    "cane_user",
]

CLASS_TO_ID = {name: idx for idx, name in enumerate(CANONICAL_CLASSES)}

SOURCE_DATE = "2026-04-20"

MOT17_EVAL_SEQUENCES = (
    "MOT17-02-FRCNN",
    "MOT17-05-FRCNN",
    "MOT17-09-FRCNN",
    "MOT17-11-FRCNN",
)

CROSSROAD_CLASS_MAPPING = {
    "pedestrian": "person",
    "person": "person",
    "wheelchair": "wheelchair_user",
    "rollator": "walker_user",
    "rollator/walker": "walker_user",
    "walker": "walker_user",
    "crutches": "crutch_user",
    "crutch": "crutch_user",
    "walking cane": "cane_user",
    "cane": "cane_user",
}

PMMA_IGNORE_KEYWORDS = (
    "empty wheelchair",
    "empty_wheelchair",
    "pusher",
    "group",
)


@dataclass(frozen=True)
class SourceDefinition:
    name: str
    folder: str
    official_url: str
    role: str
    access: str
    license_note: str
    notes: str


SOURCE_CATALOG = [
    SourceDefinition(
        name="Crossroad Mobility Aid Users",
        folder="crossroad_mobility",
        official_url="https://repository.tugraz.at/records/2gat1-pev27",
        role="Main mobility-aid detection dataset",
        access="Public repository download",
        license_note="CC BY-SA 4.0",
        notes="Use as the primary source for wheelchair, walker, crutch, and cane user boxes.",
    ),
    SourceDefinition(
        name="PMMA",
        folder="pmma",
        official_url="https://github.com/DatasetPMMA/PMMA",
        role="Mobility-aid video dataset for temporal behavior and clip evaluation",
        access="Official GitHub repo plus linked data assets",
        license_note="Verify the official PMMA data terms before redistribution",
        notes="Use for clip-based review, video sequences, and mobility-aid temporal behavior.",
    ),
    SourceDefinition(
        name="CrowdHuman",
        folder="crowdhuman",
        official_url="https://www.crowdhuman.org/",
        role="Crowded person detection and occlusion robustness",
        access="Official public website",
        license_note="Official site states non-commercial research / educational usage",
        notes="Use only for the person class, not mobility-aid classes.",
    ),
    SourceDefinition(
        name="MOT17",
        folder="mot17",
        official_url="https://motchallenge.net/data/MOT17/",
        role="Tracker tuning and held-out evaluation",
        access="Official benchmark download",
        license_note="Follow MOTChallenge terms and citation requirements",
        notes="Keep whole sequences intact; never split individual frames across splits.",
    ),
    SourceDefinition(
        name="Market-1501",
        folder="market1501",
        official_url="https://zheng-lab-anu.github.io/Project/project_reid.html",
        role="ReID pretraining",
        access="Official project page",
        license_note="Follow the official dataset terms and citation guidance",
        notes="Use unchanged for ReID pretraining, with only metadata files added if needed.",
    ),
]
