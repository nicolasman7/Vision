"""Detect + track players and the ball in a clip (YOLO + built-in ByteTrack).

Output is one row per tracked box per frame. Field Surveyor reads these parquets,
so the column list below is a contract: add columns only at the end, never rename.
"""
import pandas as pd
from ultralytics import YOLO

COLUMNS = ["frame", "track_id", "cls", "conf", "x1", "y1", "x2", "y2"]
PERSON, BALL = 0, 32  # COCO class ids


def track_clip(path, weights, imgsz=640, conf=0.1, device="mps"):
    """Track persons + sports balls through a video, streaming frame by frame.

    conf is low on purpose: ByteTrack uses low-score boxes (0.1-0.25) only to
    extend existing tracks, which helps keep IDs through pile-ups. New tracks
    still need >= 0.25 (bytetrack.yaml defaults).
    """
    model = YOLO(weights)  # fresh model per clip, so tracker state never leaks between clips
    rows = []
    results = model.track(path, tracker="bytetrack.yaml", persist=True, stream=True,
                          classes=[PERSON, BALL], imgsz=imgsz, conf=conf, device=device, verbose=False)
    for i, r in enumerate(results):
        b = r.boxes
        if b.id is None:  # no confirmed tracks in this frame
            continue
        for tid, c, s, xyxy in zip(b.id.int().tolist(), b.cls.int().tolist(), b.conf.tolist(), b.xyxy.tolist()):
            rows.append((i, tid, c, s, *xyxy))
    return pd.DataFrame(rows, columns=COLUMNS).astype({"frame": "int32", "track_id": "int32", "cls": "int8"})


def clip_stats(df, n_frames, fps):
    """Summary numbers for one tracked clip. Frames with no boxes count as zero persons."""
    persons = df[df["cls"] == PERSON]
    per_frame = persons.groupby("frame").size().reindex(range(n_frames), fill_value=0)
    lengths = df.groupby("track_id").size()  # frames each ID was seen in
    n_ids = df["track_id"].nunique()
    return {
        "persons_mean": float(per_frame.mean()),
        "persons_median": float(per_frame.median()),
        "ball_frame_rate": df.loc[df["cls"] == BALL, "frame"].nunique() / n_frames,
        "n_ids": int(n_ids),
        "median_track_len": float(lengths.median()) if n_ids else 0.0,
        "ids_per_100_frames": 100 * n_ids / n_frames,  # fragmentation: lower = steadier IDs
        "seconds": n_frames / fps,
    }
