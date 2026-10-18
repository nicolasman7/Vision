# CLAUDE.md: Vision project memory

**Read this whole file at the start of every session, before doing anything else.**

## Standing rules

1. **End of every session:** update "Current state", add a new entry at the top of "Session log" (what was done, what changed, what's broken or unresolved, where we left off), **and** update `docs/PROGRESS.md` (Gantt chart and a plain-English entry for each new item).
2. **Never `git push`** unless the user explicitly says to push in that message. Local commits are fine when asked.
3. **Never open, read, cat, grep, or print the `.env` file** (or any `.env.*`). If a value from it is needed, ask the user.
4. **Never commit videos, datasets, or model weights.** NFL footage is copyrighted (personal use only). Check `git status` before any commit.
5. `docs/PROGRESS.md` is written **for the user** (4th-year ECE student who wants to understand the project). The Gantt chart shows only things that are done or in progress, never future plans. Use real dates. Explain each concept simply, with a concrete example.
6. Don't build ahead of what's asked. The user decides when each stage starts.

## Goal

An app that watches American football film, understands what is happening, and predicts the play before the snap (run/pass, direction, depth).

- **Any angle:** All-22, broadcast, sideline, end zone, a phone filming a TV, or live from the stands. With partial visibility it still gives a best guess **plus a confidence level**.
- **Three inputs:** upload a video file, record with the camera, or a live camera feed.
- **On-device only.**

## Architecture: video → positions → prediction

```
frames
 → view_classification  which camera view is this; picks the mapping strategy
 → detection            YOLO: boxes for players + ball
 → tracking             ByteTrack: stable IDs across frames
 → field_mapping        OpenCV homography from yard lines / hash marks → field coords (yards)
 → formation            read offensive/defensive alignment at the pre-snap moment
 → prediction           run/pass, direction, depth, with confidence
```

The handoff between "vision" (first four stages) and "football" (last two) is a **per-frame list of player positions in field coordinates**, the same shape as NFL Big Data Bowl tracking data.

## Stack

- Python: Ultralytics YOLO, ByteTrack, OpenCV, training on Big Data Bowl + nflverse.
- coremltools: PyTorch → Core ML (`.mlpackage`).
- Swift/SwiftUI: iOS app (camera, upload, live overlay); macOS possibly later.
- Dev machine: M2 Pro, 16 GB RAM (train with PyTorch MPS backend).

## Folder map

| Path | Purpose |
|---|---|
| `pipeline/view_classification/` | Classify camera angle/view type |
| `pipeline/detection/` | YOLO player + ball detection |
| `pipeline/tracking/` | ByteTrack multi-object tracking |
| `pipeline/field_mapping/` | Line/hash detection, homography, image → field coords |
| `pipeline/formation/` | Formation recognition from field positions |
| `pipeline/prediction/` | Play prediction model (trained on BDB + nflverse) |
| `pipeline/export/` | coremltools conversion code |
| `data/raw/`, `data/processed/` | Datasets (git-ignored) |
| `models/` | Trained weights (git-ignored) |
| `coreml/` | Core ML exports consumed by the app (git-ignored) |
| `app/` | Xcode project (iOS, maybe macOS) |
| `notebooks/` | Exploration only; real logic moves into `pipeline/` |
| `scripts/` | CLI entry points: download, train, evaluate, export |
| `tests/` | Python tests for `pipeline/` (Swift tests live in the Xcode project) |
| `docs/` | Docs; `PROGRESS.md` is the user's learning log |

## Key decisions and reasoning

- **Field coordinates as the shared representation.** Each camera angle looks different, but a player at the 30-yard line on the left hash is in the same place on the field no matter where the camera is. Converting everything to top-down yards splits a hard problem in two: vision (angle-specific) and prediction (angle-agnostic). It also means the predictor can be trained on Big Data Bowl tracking data, which is already in field coordinates, without needing labeled video.
- **View classification first.** All-22 shows the whole field; broadcast shows part of it; a phone at a TV adds skew and glare. Knowing the view tells field_mapping which landmarks to expect and how much to trust the result.
- **Confidence is part of the output, not optional.** Partial views are the normal case (most broadcast shots cut off part of the field). Each stage carries uncertainty forward: missed players, a shaky homography, an ambiguous formation. The final prediction reports it.
- **Tracking-by-detection with ByteTrack.** It needs no extra re-identification network, runs cheaply, and keeps low-confidence boxes, which helps when players overlap in a crowd.
- **Monorepo.** Python produces the models the Swift app loads. Keeping both in one repo means model and app changes stay in sync.
- **Notebooks are scratch space.** Anything that works gets moved into `pipeline/` so it can be tested and exported.
- **Data, weights, and exports are git-ignored.** They are large, can be regenerated, and the footage is copyrighted.

## Constraints

- On-device inference only (Core ML / Neural Engine). Models must be small enough for real-time use on iPhone. Training must fit in 16 GB of unified memory.
- Must handle any camera angle and degrade gracefully, giving a lower-confidence guess rather than no answer.
- NFL footage is copyrighted: personal use only, never commit videos or frames.

## Open questions (not decided yet)

- Python environment/package manager (uv, conda, venv).
- Minimum iOS version; whether macOS is a target.
- Labeled football detection data source (public datasets vs. hand-labeling).
- Exact prediction targets and buckets for "direction" and "depth".

## Current state

_As of 2026-10-17_

- Project structure and docs created. `git init` done; **no commits yet**.
- No code, no environment, no dependencies, no data downloaded.
- Next step: up to the user.

## Session log

### 2026-10-17: Project setup
- **Done:** Created folder layout (`pipeline/` with 7 stage sub-folders, `data/`, `models/`, `coreml/`, `app/`, `notebooks/`, `scripts/`, `tests/`, `docs/`), a README in each top-level folder, root README, `.gitignore`, this file, and `docs/PROGRESS.md`. Ran `git init`.
- **Changed:** Everything is new. Empty pipeline sub-folders and `data/raw`, `data/processed` hold `.gitkeep` files so git tracks them.
- **Broken/unresolved:** Nothing broken. Open questions listed above. Nothing committed yet.
- **Left off:** Waiting for the user to choose the first stage to build.
