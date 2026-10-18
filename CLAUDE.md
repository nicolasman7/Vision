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
| `docs/research/` | Option docs for the user's pending decisions (agents recommend, the user decides) |
| `tools/agent-hq/` | Agent HQ: live game dashboard of subagents (`report.py` writes status, `index.html` shows it) |

## Agent team

The user wants Claude to act as **lead ("Coach")** and delegate to labeled, narrow-niche subagents, shown live in Agent HQ. Every subagent reports progress with:
`python3 tools/agent-hq/report.py <id> "<Name>" "<Role>" <working|done|blocked|idle> "<task>"`
Roles map to game stations: Lead → Office, Data → Data Mine, ML → Training Gym, Docs → Library, UI → Workshop, Vision → Vision Lab (any other role also goes to the Vision Lab desk). Roster: `lead` Coach (Lead), `strategist` Strategist (Strategy; a Plan agent that designs each round's team), `wrangler` Data Wrangler (Data), `modeler` Modeler (ML), `bucketeer` Bucket Scout (ML), `surveyor` Field Surveyor (Vision), `scout` Film Scout (Docs, web research), `teacher` Teacher (Docs, writes PROGRESS.md), `gamedev` Game Dev (UI). The roster panel has Active and Bench tabs. A round starts with Strategist planning; the lead then launches agents in phases, with file ownership per agent and only the lead running `uv add`. Agents that finish aren't deleted: they go back on the **bench** (`idle`, "On the bench: …") as reserves, and the lead reuses an existing agent before creating a new one with an overlapping niche. The lead reviews every subagent's output (runs tests, spot-checks) before handing off to the next. View the game: `python3 -m http.server 8765 -d tools/agent-hq`, then open http://localhost:8765.

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

- Minimum iOS version; whether macOS is a target.
- Labeled football detection data source (public datasets vs. hand-labeling). Evidence: `docs/research/detection_datasets.md`.
- Exact prediction targets and buckets for "direction" and "depth". Evidence: `docs/research/direction_depth_options.md` (recommends D2 runs-only L/M/R; P3 ≤0/1–9/10+ on throws).
- Where the app gets game situation (down, distance, clock, score): scorebug OCR vs. user input. Evidence: `docs/research/situation_source.md` (E4: no scoreboard 75.0% vs full 76.2%; the scoreboard mostly sharpens confidence).

## Current state

_As of 2026-10-18 (session 4)_

- `git init` done; **no commits yet**. A pre-round-2 code backup tarball was made in the session scratchpad (temporary).
- Python env: **uv**, Python 3.12. Deps: kaggle, pandas, pyarrow, scikit-learn, opencv-python-headless, numpy; dev: pytest. `uv run pytest` → 14 passing.
- Data: nflverse pbp + participation 2016–2025 and FTN charting 2022–2025 in `data/raw/nflverse/` (~224 MB). BDB 2025 **not downloaded** (needs the user's Kaggle token); `bash scripts/download_data.sh --bdb` runs that part.
- Prediction: `scripts.build_dataset` → `data/processed/plays_<year>.parquet` + `plays_2016_2025.parquet` (342,499 plays, 45 cols; era-specific columns documented in `dataset.py`). `scripts.train_prediction` = original 2022 model (75.2%). `scripts.experiments_prediction` = E1–E5 → `models/metrics_multiseason.json`, `models/run_pass_multiseason.joblib` (2025 test: 76.2% vs 60.4% always-pass). `scripts.analyze_buckets` → `models/metrics_buckets.json`. Direction has no real skill without x/y.
- Field mapping v0: `pipeline/field_mapping/` (geometry, homography fit with float64 DLT refit, synthetic cameras); `scripts.field_mapping_error_budget` prints the error table. No pixel line detection yet (needs footage).
- Agent HQ: see "Agent team". Bench: wrangler, modeler, teacher, gamedev, strategist, surveyor, scout, bucketeer.

## Session log

### 2026-10-18: Round 2 (Strategist + 6 agents)
- **Done:** Strategist (Plan agent) designed a 3-phase plan. Phase 1: Data Wrangler extended the data to 2016–2025 + FTN (era harmonization, FTN leak ban with a test); Field Surveyor built field_mapping v0 + error budget; Film Scout wrote the detection-data and situation-source docs. Phase 2: Modeler ran E1–E5 (added `season_split`, `corrupt_category`, `shift_count`, `ERA_FEATURES`; additions only); Bucket Scout compared direction/depth bucket schemes. Phase 3: Teacher wrote the PROGRESS entry and filled the E4 section of situation_source.md. Coach added OpenCV, the "Vision Lab" station (role `vision`), Active/Bench tabs (via Game Dev), and reviewed every handoff.
- **Changed:** `download_data.sh` runs BDB only with `--bdb`. FB counts as RB. Box counts stay in per-era columns (`defenders_in_box`, `box_part23`, `ftn_box`), never merged. `offense_formation` is NaN for 2023+; use `formation_family`/`is_empty` across eras.
- **Broken/unresolved:** Nothing broken. Findings: more seasons +0.1–0.2 pts only; FTN columns add ~0; each 10% formation misread costs ~1.5 pts; the 2025 model is 1–2 pts overconfident in the 60–90% bins; 2024 pistol spike (9.3%) unexplained; formation-dropout training (to handle unreadable formations) suggested but not built. Nothing committed.
- **Left off:** Waiting on the user: first commit, Kaggle token, 3 decisions (docs/research), footage clips for vision, approval of heavy deps (torch/ultralytics) when detection starts.

### 2026-10-17: Agent team + first play predictor
- **Done:** Set up a lead + subagent workflow and the Agent HQ status system. Data Wrangler built `pipeline/prediction/dataset.py`, `scripts/build_dataset.py`, and tests (label = `qb_dropback` intent; drops no_play/2-pt/aborted; parses personnel; `off_ol` defaults to 5). Modeler built `pipeline/prediction/model.py`, `scripts/train_prediction.py`, and tests (time split by week ≤14 vs >14; HistGradientBoosting; isotonic calibration tried, made it worse, not used). Teacher wrote the PROGRESS.md entry. Game Dev built `tools/agent-hq/index.html`.
- **Changed:** Pivoted v1 inputs from BDB x/y positions to nflverse participation (formation/personnel labels), so no Kaggle signup was needed. Added scikit-learn and pytest.
- **Broken/unresolved:** Direction has no skill without real positions (needs BDB tracking or vision output). The model leans on down/distance/clock/score, which vision can't see (new open question). Only one season. Nothing committed.
- **Left off:** The user decides next. Options: BDB positions for direction, more seasons, depth buckets, or starting the vision side.

### 2026-10-17: uv environment + nflverse data
- **Done:** `uv init --bare`, Python 3.12 pinned, added kaggle/pandas/pyarrow. Downloaded nflverse `play_by_play_2022.parquet` (49,434 plays × 372 cols; 20,393 pass, 15,037 run; has `run_location`, `run_gap`, `pass_location`, `air_yards`). Wrote `scripts/download_data.sh`.
- **Changed:** New `pyproject.toml`, `uv.lock`, `.python-version`, `scripts/download_data.sh`. Settled the open question on the Python env manager (uv).
- **Broken/unresolved:** BDB 2025 needs the user to accept the competition rules and put an API token in `~/.kaggle/kaggle.json` (never read that file). Nothing committed.
- **Left off:** Once the token exists, run `bash scripts/download_data.sh`, then explore and join BDB plays with nflverse on (gameId, playId) ↔ (old_game_id, play_id).

### 2026-10-17: Project setup
- **Done:** Created folder layout (`pipeline/` with 7 stage sub-folders, `data/`, `models/`, `coreml/`, `app/`, `notebooks/`, `scripts/`, `tests/`, `docs/`), a README in each top-level folder, root README, `.gitignore`, this file, and `docs/PROGRESS.md`. Ran `git init`.
- **Changed:** Everything is new. Empty pipeline sub-folders and `data/raw`, `data/processed` hold `.gitkeep` files so git tracks them.
- **Broken/unresolved:** Nothing broken. Open questions listed above. Nothing committed yet.
- **Left off:** Waiting for the user to choose the first stage to build.
