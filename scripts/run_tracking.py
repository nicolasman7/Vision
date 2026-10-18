"""Track every clip in data/raw/clips/ with 2 model sizes x 2 input sizes; keep the best.

    uv run python -m scripts.run_tracking            # all clips, all settings
    uv run python -m scripts.run_tracking --new      # only clips not in results.json yet

Writes (all git-ignored):
  data/processed/tracks/<clip_id>.parquet          best setting's tracks
  data/processed/vision/spotter/<clip_id>_fNNNN.jpg  annotated frames (boxes + IDs)
  data/processed/vision/spotter/results.json       every setting's FPS + stats
"""
import json
import sys
import time
from pathlib import Path

import cv2

from pipeline.tracking.track import BALL, clip_stats, track_clip

CLIPS = Path("data/raw/clips")
MANIFEST = Path("docs/clips_manifest.json")
TRACKS = Path("data/processed/tracks")
OUT = Path("data/processed/vision/spotter")
# Time-boxed run: one setting. Widen to ["yolo26n", "yolo26s"] x [640, 1280] to compare.
MODELS = ["yolo26s"]
SIZES = [1280]  # players are tiny in wide shots; 640 shrinks them below what YOLO sees
N_SHOTS = 2


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return n, fps


def frames_at(path, wanted):
    """Yield (index, frame) for the wanted indices, reading the video once as a stream."""
    wanted = set(wanted)
    cap = cv2.VideoCapture(str(path))
    i = 0
    while wanted:
        ok, img = cap.read()
        if not ok:
            break
        if i in wanted:
            wanted.discard(i)
            yield i, img
        i += 1
    cap.release()


def draw(img, boxes):
    for r in boxes.itertuples():
        color = (0, 165, 255) if r.cls == BALL else (0, 255, 0)
        p1, p2 = (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2))
        cv2.rectangle(img, p1, p2, color, 1)
        cv2.putText(img, str(r.track_id), (p1[0], p1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return img


def pick_best(runs):
    # Recall proxy: most persons per frame. Fans/sideline also count, so the table and the
    # annotated frames are the real judge; this rule only picks what gets saved.
    return max(runs, key=lambda r: (round(r["persons_mean"], 1), -r["ids_per_100_frames"]))


def run_clip(path, angle):
    n_frames, fps = video_info(path)
    runs, best_df = [], None
    for m in MODELS:
        for sz in SIZES:
            t = time.perf_counter()
            df = track_clip(str(path), f"models/yolo/{m}.pt", imgsz=sz)
            # Wall time includes model load + MPS warm-up; small next to ~900 frames.
            run = {"clip": path.stem, "angle": angle, "model": m, "imgsz": sz,
                   "fps": n_frames / (time.perf_counter() - t), **clip_stats(df, n_frames, fps)}
            print({k: round(v, 3) if isinstance(v, float) else v for k, v in run.items()}, flush=True)
            runs.append(run)
            if pick_best(runs) is run:
                best_df = df
    best = pick_best(runs)
    best["best"] = True
    TRACKS.mkdir(parents=True, exist_ok=True)
    best_df.to_parquet(TRACKS / f"{path.stem}.parquet", index=False)

    OUT.mkdir(parents=True, exist_ok=True)
    shots = [int(n_frames * (k + 0.5) / N_SHOTS) for k in range(N_SHOTS)]
    for i, img in frames_at(path, shots):
        cv2.imwrite(str(OUT / f"{path.stem}_f{i:04d}.jpg"), draw(img, best_df[best_df["frame"] == i]))
    return runs


def main():
    angles = {c["id"]: c["angle"] for c in json.loads(MANIFEST.read_text())}
    clips = sorted(p for p in CLIPS.glob("*.mp4") if not p.name.startswith("."))
    results_path = OUT / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for c in clips:
        if "--new" in sys.argv and c.stem in results:  # only clips not processed yet
            continue
        results[c.stem] = run_clip(c, angles.get(c.stem, "unknown"))
        results_path.write_text(json.dumps(results, indent=1))

    print("\n| clip | angle | model | imgsz | FPS | persons/frame (mean/med) | ball rate | IDs | median track len | IDs/100 fr |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for runs in results.values():
        for r in runs:
            star = " *" if r.get("best") else ""
            print(f"| {r['clip']} | {r['angle']} | {r['model']}{star} | {r['imgsz']} | {r['fps']:.1f} | "
                  f"{r['persons_mean']:.1f}/{r['persons_median']:.0f} | {r['ball_frame_rate']:.2f} | "
                  f"{r['n_ids']} | {r['median_track_len']:.0f} | {r['ids_per_100_frames']:.1f} |")


if __name__ == "__main__":
    main()
