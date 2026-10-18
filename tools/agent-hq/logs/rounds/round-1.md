# Round 1: first play predictor + Agent HQ

**Date:** 2026-10-17 (about 23:25 to 23:45 EDT). Written after the fact on 2026-10-18 from the agents' final reports; the automatic activity log started in round 3.
**Goal:** train a run/pass predictor from free nflverse data (no Kaggle), and build a live dashboard of the agents.

## Coach (lead)
- **Job:** wrote `tools/agent-hq/report.py` (the status protocol) and git-ignored `status/`.
- **Hand-offs:** Data Wrangler → Modeler → Teacher, with Game Dev in parallel.
- **Review:** reviewed every hand-off (tests, leak check, git-ignore check).

## Data Wrangler (Data)
- **Job:** joined 2022 play-by-play with participation data (formation and personnel).
- **Result:**
  - 49,434 → 35,193 plays after filtering; 100% join; 60.4% pass.
  - The label is `qb_dropback`, which captures the play call: sacks and scrambles still count as passes.
  - Personnel strings were parsed with a regex. `off_ol` defaults to 5.
  - Post-snap columns are banned.
- **Files:** `pipeline/prediction/dataset.py`, `scripts/build_dataset.py`, `tests/test_prediction_dataset.py`, plus the participation download in `scripts/download_data.sh`.
- **Lead review:** tests pass, no duplicate plays. EMPTY formation is 97% pass, I_FORM 28%.

## Modeler (ML)
- **Job:** trained the run/pass and direction models, split by time (train weeks 1–14, test weeks 15–22).
- **Result (accuracy on the test weeks):**
  - always-pass 59.4%
  - situation only 70.7%
  - situation + formation 75.2%, well calibrated (every bin within about 1 point); isotonic calibration made it worse
  - direction 40.1% vs 40.2% baseline, so no skill
  - top feature: `offense_formation`
- **Files:** `pipeline/prediction/model.py`, `scripts/train_prediction.py`, `tests/test_prediction_model.py`. Added scikit-learn.
- **Lead review:** 4 tests pass, labels aren't used as features, `models/` is git-ignored.

## Teacher (Docs)
- **Job:** wrote the "First play predictor" entry in `docs/PROGRESS.md`, covering leakage, time splits, baselines, boosted trees, calibration, log loss and permutation importance.

## Game Dev (UI)
- **Job:** built `tools/agent-hq/index.html`: a pixel-art football field with stations, a bench, a trophy zone and a roster panel.
- **Result:**
  - Tested with headless Chrome screenshots.
  - Later called off the bench to add Active/Bench tabs and space out the bench seats.

## Waiting on the user after round 1
- the Kaggle token
- the direction and depth bucket choices
- the first commit
