# Changelog

Every app version, newest first. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Dates are real.

How we tried to improve the model itself: [research/model_improvement.md](research/model_improvement.md). The plain-English learning log: [PROGRESS.md](PROGRESS.md).

## [3.0] - 2026-10-19

### Added
- **Guess → reveal → learn** in Play lookup: the app makes its call, you reveal what really happened, and it learns from the answer.
- **On-device online correction layer:** a 16-feature logistic regression on top of the frozen v3 tree model. It starts as "trust the base model" and takes one SGD step per revealed play (learning rate 0.003, L2 0). Its settings were picked in an offline simulation (see model_improvement.md).
- **"Why it missed" factors:** for each group of inputs (down & distance, formation, box…), how much it moved the pass probability.
- **Learning card:** base vs adjusted accuracy and log loss, scored prequentially (each guess is graded before the model learns from it). Includes a **Reset learning** button.

### Changed
- App version is stamped as 3.0.

### Known limitations
- In simulation the correction layer helps only slightly: log loss 0.4707 → 0.4679 over 69,201 plays, and no measurable change over the first 50–200 plays.
- Learning stays on one device and isn't backed up. Reset erases it.

## [2.0] - 2026-10-19

### Added
- **Live / Record / Upload camera** with on-device YOLO player boxes (yolo26n converted to Core ML, 4.8 MB).
- **Play lookup tab** over 342,499 plays (2016–2025), stored in an on-device SQLite database (55 MB, not in git).
- **Defense inputs:** personnel presets and box count.
- A UI test for the checkmark fix.

### Changed
- **Apple Sports-style redesign:** field diagram, your favorite accent color applied to all text, and an in-app light / dark / system setting.
- numpy pinned to ≤2.3.5 so coremltools works.

### Fixed
- Checkmark bug.

### Known limitations
- YOLO boxes are drawn on screen but don't feed the predictor yet. Formation still comes from the input panel.
- The Play lookup database has to be built locally (`scripts/build_history_db.py`).

## [1.0] - 2026-10-19

### Added
- **iPhone app shell:** SwiftUI input panel and result card (run/pass, run direction, pass depth).
- The 3 Core ML models (run/pass, direction, depth) and `predictor_schema.json`, bundled in the app.
- Swift ↔ Python parity tests: 21 fixtures, max difference 1.1e-16 on macOS.

### Known limitations
- Inputs are typed in by hand. No camera.

## Model versions

An **app version** is the whole iPhone app: screens, camera, database. A **model version** is the trained predictor inside it. They change independently: app v2.0 and v3.0 both ship the same v3 model.

| Model | Trained on | Test result | What's new |
|---|---|---|---|
| v1 | 2022 weeks 1–14 | 75.2% on 2022 weeks 15–22 | First run/pass model (situation + formation) |
| multiseason | 2016–2024 | 76.2% on 2025 | Ten seasons, formation labels that work across eras |
| v3 bundle | 2016–2023 (calibrated on 2024) | 76.2% on 2025 with all inputs, 74.9% with no situation | Input dropout (one model for any missing inputs) + 3 heads (run/pass, run direction, pass depth) |
