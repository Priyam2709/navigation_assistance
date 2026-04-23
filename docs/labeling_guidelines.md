# Labeling Guidelines

Use these rules whenever you review mobility-aid data in CVAT.

## Canonical classes

The final class map for v1 is:

1. `person`
2. `wheelchair_user`
3. `walker_user`
4. `crutch_user`
5. `cane_user`

## What each class means

- `person`: a pedestrian without a visible mobility aid.
- `wheelchair_user`: a person seated in or clearly using a wheelchair.
- `walker_user`: a person using a walker or rollator.
- `crutch_user`: a person using one or two crutches.
- `cane_user`: a person using a walking cane.

## Bounding-box rule

Draw a single full-body box around the passenger.

When the aid is visually attached to the person’s silhouette, include it inside the same box.

Examples:

- include the wheelchair frame with the seated user
- include the walker with the person using it
- include the crutch or cane if it is clearly part of the user pose

## Ignore rule

Mark the annotation as `ignore` if any of these are true:

- the subject is tiny and not usable for training
- the image is heavily blurred
- the person is mostly hidden, around more than `70%` occluded
- it is a reflection, poster, mannequin, or screen image
- it is an empty wheelchair without a clearly associated seated user
- it is only the pusher of a wheelchair and not the assisted user
- it is a large group box that merges multiple people into one annotation

## PMMA-specific review rule

For v1, keep only boxes that cleanly map to the canonical class set.

Mark these as `ignore`:

- empty wheelchair
- pusher-only box
- mixed group box combining pusher and wheelchair occupant

## Track IDs for video clips

If you are reviewing video or clip frames, keep a stable `track_id` for the same passenger within a clip.

Do not reuse the same `track_id` for a different passenger in the same clip.

## CVAT checklist

Before exporting:

1. confirm the class name matches the canonical list
2. confirm the box covers the user, not just the aid
3. confirm `ignore` is set for unusable cases
4. confirm `track_id` stays stable across frames for the same person

## Export convention

Place reviewed exports here:

```text
data/working/cvat/reviewed/crossroad_mobility.jsonl
data/working/cvat/reviewed/pmma.jsonl
```

The build script automatically prefers these reviewed files over the raw converted manifests if they exist.
