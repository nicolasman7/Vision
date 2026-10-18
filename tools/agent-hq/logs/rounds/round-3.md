# Round 3: app-ready predictor, Core ML, first real-footage vision

**Date:** 2026-10-18 (00:20–02:30 EDT). An API session limit paused the vision agents for about an hour mid-round.
**Goal:** make the predictor app-ready and start the vision side on openly licensed footage. No Kaggle.

## Coach (lead)
- **Installs:** torch 2.14.1 (MPS), ultralytics 8.4.182, coremltools 9.0, and later `lap`. opencv-python-headless was swapped for opencv-python because ultralytics conflicts with it.
- **Setup:** turned off Ultralytics telemetry, recorded the user's decisions in CLAUDE.md, and added the delegation-reminder hook.
- **Calls made:**
  - Shipped run/pass with identity calibration, as Modeler recommended.
  - Removed Spotter's temporary stand-in for `lap`.
- **Review:** checked every hand-off (tests, parity re-run, sanity predictions).

## Strategist: plan
Planned 6 agents in 4 phases. Before planning, it verified two things:
- The sklearn→Core ML converter is disabled for sklearn 1.9, so trees must be converted directly.
- Licensed clip sources stream over HTTP, so clips can be cut without downloading whole games.

## Modeler (ML): predictor v3
- **What it built:** one bundle with three heads (run/pass, run direction, throw depth), trained with input-block dropout.
- **Results on 2025:**
  - Run/pass with full inputs: 76.2%. Every input-availability pattern is within about 0.25 points of a model trained for that pattern.
  - Blanking an unsure formation is better than guessing it: at a 30% error rate, blank costs 0.8 points and misread costs 4.4.
  - Direction: temperature T = 1.24. Depth: T = 1.00.

## Core ML Packer (ML), new
- **What it built:** `pipeline/export/hgb_to_coreml.py`, which converts the fitted trees straight into Core ML.
- **NaN check:** a spike test proved Core ML handles NaN natively.
- **Parity vs Python, on 2,000 plays from 2025:** worst difference 3.3e-16 on all three heads.
- **Size:** about 0.9 MB in total.

## Film Scout (Docs): clips
- **Clips:** 3 openly licensed high-school clips from the Internet Archive, cut by streaming (no full downloads).
  - 2 sideline-elevated (CC0)
  - 1 high-wide (Public Domain Mark)
- **Records:** a license manifest and `scripts/fetch_clips.py`.
- **Still missing:** end zone, All-22, stands, and phone-at-TV (only the user can film that).

## Spotter (Vision), new
- **What it ran:** YOLO26s + ByteTrack at imgsz 1280, 24–27 FPS on MPS.
- **Failure modes:**
  - Sideline people are counted as players (about half of the detections).
  - The ball is seen in only 2–13% of frames.
  - IDs fragment in pile-ups.

## Field Surveyor (Vision)
- **What it built:**
  - `lines.py`: grass mask → white paint → Hough lines → landmarks.
  - A per-frame homography with failure reasons.
  - The first top-down demo.
- **Mappable rate:** high-wide 8%, oblique sideline 0%.
- **Limits:** x is relative (an unknown multiple of 5 yd), and drift isn't measured.

## Teacher (Docs)
- **What it wrote:** the PROGRESS.md entry and a recruiter-facing README.

## Waiting on the user
- **Footage only the user can provide:** a phone-at-TV clip and a true All-22 clip.
- **Labeling:** choose the label classes and hand-label frames when ready.
- **iOS target version:** still open.
- **Test on a real iPhone:** re-check Core ML parity once the Swift app exists.
