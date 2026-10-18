# Vision

Vision watches American football film, works out what is happening on the field, and predicts the play before the snap.

## What it does

1. **Takes video in three ways:** upload a file, record with the camera, or point the camera at a live feed (a TV or a real field).
2. **Works from any angle:** All-22 overhead, broadcast TV, sideline, end zone, a phone filming a TV, or a seat in the stands. When it can't see everything, it still gives its best guess along with a confidence level.
3. **Understands the field:** finds the players and the ball, tracks them from frame to frame, and places them on a top-down field using the yard lines and hash marks it can see.
4. **Reads the formation and predicts the play:** run or pass, which direction, and how deep.

Everything runs on the device. Nothing is sent to a server.

## How it works

```
video frames
   → what kind of camera view is this?          (view classification)
   → where are the players and the ball?        (detection)
   → which box is the same player over time?    (tracking)
   → where is each player on the real field?    (field mapping / homography)
   → what formation is the offense in?          (formation)
   → what play is coming?                       (prediction)
```

The key idea: every camera angle is turned into the **same thing**, a list of player positions in field coordinates (yards). That matches the format of the NFL Big Data Bowl tracking data, so the prediction model can be trained on that data and then run on positions taken from any video.

## Stack

- **Python** for the ML pipeline: YOLO (detection), ByteTrack (tracking), OpenCV (homography), and models trained on NFL Big Data Bowl tracking data plus nflverse play-by-play.
- **coremltools** to convert the trained models to Core ML.
- **Swift / SwiftUI** for the iOS app (camera, upload, live overlay), and possibly macOS.

## Folder map

| Folder | What's in it |
|---|---|
| `pipeline/` | Python ML code, one sub-folder per stage |
| `data/` | Datasets, raw and processed (git-ignored) |
| `models/` | Trained model weights (git-ignored) |
| `coreml/` | Core ML exports for the app (git-ignored) |
| `app/` | The Xcode project for iOS (and maybe macOS) |
| `notebooks/` | Jupyter notebooks for exploring ideas |
| `scripts/` | One-off command-line tools: downloading data, training, exporting |
| `tests/` | Python tests for the pipeline |
| `docs/` | Documentation, including `PROGRESS.md` |

## Legal note

NFL footage is copyrighted. This project is for personal use only. Videos are never committed to the repository.

## Status

Project setup only. No code yet. See [docs/PROGRESS.md](docs/PROGRESS.md).
