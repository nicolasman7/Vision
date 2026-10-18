# Round 2: ten seasons, field mapping, decision research

**Date:** 2026-10-17 23:40 to 2026-10-18 00:05 EDT. Written after the fact on 2026-10-18 from the agents' final reports.
**Goal:** tackle four issues without anything that needs the user:
- direction has no skill
- only one season of data
- depth isn't modeled
- the vision side hasn't started

## Strategist (Strategy, planning only)
- **Job:** checked what data exists and designed a 6-agent, 3-phase plan with file ownership.
- **What it found:**
  - Participation data exists for 2016–2025, FTN charting for 2022–2025.
  - Formation labels change in 2023 (only SHOTGUN, UNDER CENTER and PISTOL), and personnel becomes position-level.
  - `time_split` breaks once several seasons are stacked.
  - Without x/y positions, direction has a hard ceiling.

## Coach (lead), phase 0
- Backed up the code.
- Installed `opencv-python-headless` and `numpy`.
- Added the "Vision Lab" station (role `vision`) to Agent HQ.
- Reviewed each phase before starting the next.

## Phase 1 (in parallel)

### Data Wrangler (Data)
- **Result:**
  - 342,499 plays covering 2016–2025.
  - Harmonized `formation_family` and `is_empty`; box counts kept per era (NGS and FTN agree on only 59% of plays).
  - FTN post-snap columns are banned, with a test.
  - 2022 still reproduces 35,193 rows and 0.7523 accuracy.
- **Corrections and oddities:**
  - "OL" in 2023 was really "OLB" matched by a substring search.
  - 2024 has a pistol spike (9.3% of plays).
- **Files:** `dataset.py`, `build_dataset.py`, `download_data.sh` (the BDB part now only runs with `--bdb`), `test_prediction_dataset.py`.

### Field Surveyor (Vision)
- **Result:**
  - `pipeline/field_mapping/` v0: field geometry in Big Data Bowl coordinates and a homography fit with a float64 refit, tested on synthetic pinhole cameras.
  - With 1 px of noise, broadcast error is 0.014 yd median and 0.04 yd p95.
  - When only one line of landmarks is visible, it returns "unmappable" instead of guessing.
- **Open problems:** telling which yard line is which, lens distortion, camera motion.

### Film Scout (Docs)
- **Result:** `docs/research/detection_datasets.md` and `docs/research/situation_source.md`.
- **Findings:**
  - Every football-specific detection set needs an account.
  - Recommends a COCO YOLO start, then hand-labeling about 500 frames.
  - Ultralytics weights are licensed AGPL-3.0.

## Phase 2 (in parallel, after Data Wrangler)

### Modeler (ML)
- **Result:**
  - E1: more seasons barely help (0.7523 → 0.7535).
  - E2: the 2025 test gets 76.2% vs 60.4% always-pass.
  - E3: FTN adds about nothing.
  - E4: no scoreboard gives 75.0%, full features 76.2%.
  - E5: every 10% of formation misreads costs about 1.5 points; ±1 box errors cost nothing.
- **Files:** `model.py` (additions only), `scripts/experiments_prediction.py`, `test_prediction_model.py`.

### Bucket Scout (ML)
- **Result:**
  - No direction scheme has real skill; the best is about a 2% log-loss reduction.
  - Recommends D2 for direction (runs only, L/M/R).
  - Recommends P3 for depth (≤0 / 1–9 / 10+, throws only).
  - In 2025, accuracy overstated the skill (runs happened to swing right that year); log loss didn't.
- **Files:** `scripts/analyze_buckets.py`, `tests/test_buckets.py`, `docs/research/direction_depth_options.md`.

## Phase 3

### Teacher (Docs)
- **Result:**
  - Wrote the "Ten seasons, field mapping, and decision research" entry in PROGRESS.md.
  - Filled in the E4 section of `situation_source.md`.
  - Found that a confidently wrong formation is worse than leaving it blank.

**Lead review:** 14 tests pass, the mermaid chart is valid, and no data or weights are in git.

## Waiting on the user after round 2
- Three decisions: direction/depth buckets, the situation source, and the detection data source.
- The first commit.
- The Kaggle token.
- Footage clips.
- Approval for heavy dependencies (torch/ultralytics).
