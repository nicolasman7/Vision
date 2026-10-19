# Vision

**An on-device app that watches American football film and predicts the play before the snap: run or pass, direction, and depth, each with a confidence level.**

## The goal

- **Any camera angle:** All-22, broadcast, sideline, end zone, a phone filming a TV, or a seat in the stands. If the view is partial, it still gives a best guess and says how sure it is.
- **Three inputs:** upload a video file, record with the camera, or use a live camera feed.
- **On-device only:** Core ML on iPhone. No server, and no video ever leaves the device.

## Architecture: vision → field coordinates → prediction

```
frames → view classification → detection (YOLO) → tracking (ByteTrack)
       → field mapping (yard lines/hashes → homography → yards)
       → formation → prediction (run/pass, direction, depth + confidence)
```

Every camera angle is reduced to the same thing: player positions in field coordinates (yards), the same format as NFL tracking data. That splits one hard problem into two:
- **Vision** depends on the camera angle and turns pixels into yards.
- **Prediction** doesn't care about the angle. It's trained on ten seasons of play-by-play data, with no labeled video needed.

## Current results

Pre-snap run/pass predictor, trained on 2016–2023, calibrated on 2024, tested on the full 2025 season (34,413 plays):

| Inputs available | Accuracy |
|---|---|
| Everything (game situation + what the camera sees) | **76.2%** |
| Camera only (no scoreboard) | 74.9% |
| Situation only (vision failed) | 70.7% |
| Baseline: always guess "pass" | 60.4% |

- **One model covers every pattern of missing inputs.** It was trained with input dropout, and each pattern scores within about 0.25 points of a model trained specially for it.
- **Its confidence is honest:** each confidence band is within about 1 point of the real hit rate.
- **Two extra heads, conditional on the play type:**
  - direction for runs: 40.2% vs a 36.4% base rate
  - depth for throws: 48.6% vs 45.7%
- **Core ML export:** we wrote our own tree converter. Over 2,000 test plays, the Core ML models match Python to within 3.3e-16, and their top pick agrees 100% of the time. The models are 124–394 KB and take about 0.02 ms per prediction. So far this is tested on macOS, not yet on iPhone.
- **Vision on openly licensed high-school clips:**
  - YOLO + ByteTrack runs at 24–27 FPS on an M2 Pro.
  - Field-line detection plus a homography produces a top-down view, but only on some frames, and it's a demo, not a measurement.

The full learning log, with every experiment explained, is in [docs/PROGRESS.md](docs/PROGRESS.md).

## Tech stack

- **Python:** pandas, scikit-learn (gradient-boosted trees), Ultralytics YOLO, ByteTrack, OpenCV; managed with uv.
- **Core ML:** coremltools, plus a custom exporter from scikit-learn trees to Core ML.
- **Swift / SwiftUI:** the iOS app (in progress).
- **Data:** nflverse play-by-play and participation data (2016–2025), FTN charting (2022–2025).

## Repo map

| Path | What's in it |
|---|---|
| `pipeline/` | One folder per stage: `detection`, `tracking`, `field_mapping`, `formation`, `prediction`, `export`, … |
| `scripts/` | Command-line entry points: build dataset, train, run experiments, export, run tracking, demos |
| `tests/` | pytest suite for the pipeline (including checks that no post-snap information leaks into training) |
| `docs/` | `PROGRESS.md` learning log, `research/` decision docs, clip manifest |
| `coreml/` | `predictor_schema.json`, the input contract between vision and the predictor (model files git-ignored) |
| `app/` | Xcode project (iOS) |
| `tools/agent-hq/` | Live dashboard of the agent team |

## How it was built

The project is built with an agentic AI harness: a lead orchestrator agent coordinates a team of specialized AI subagents, each with one narrow job: a data wrangler, a modeler, a field surveyor, a film scout, a teacher who writes the docs, and others. They reported their status to a game-style live dashboard (`tools/agent-hq/`). The playbook is in [AGENT_TEAM.md](AGENT_TEAM.md). Every session is written up in plain English in [docs/PROGRESS.md](docs/PROGRESS.md).

## Roadmap

- Formation stage: read the offense's alignment from tracked players
- Separate players from sideline people, and tell offense from defense
- Read the painted yard numbers so field position is absolute, not relative
- Big Data Bowl (x, y) features to improve direction and depth
- iOS app: camera, upload, live overlay, and on-device testing

## Data & licenses

- **No datasets, footage, or model weights are in this repo.** Data comes from [nflverse](https://github.com/nflverse) (see their licenses). The video clips are openly licensed (CC0 / Public Domain Mark), and their sources are listed in [docs/clips_manifest.json](docs/clips_manifest.json).
- Ultralytics YOLO is AGPL-3.0 and is used as a dependency.
- Code: © 2026 Nicolas Mansour, all rights reserved. The source is public so it can be reviewed. See [LICENSE](LICENSE).
