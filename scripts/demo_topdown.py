"""Top-down demo: Spotter's tracks + field mapping -> player positions in (relative) yards.

Run from repo root: uv run python -m scripts.demo_topdown [clip_id]

1. Every frame: try fit_frame (landmarks -> H) and measure camera motion from the last frame
   (grass corners tracked with optical flow -> frame-to-frame homography F).
2. Anchor on the most physically plausible fitted frame and carry its H to every other frame
   through the chain of F's. One anchor keeps the relative x consistent across the clip
   (different frames would each call their leftmost visible line x = 0).
3. Feet = bottom-centre of each person box -> field yards. Writes
   data/processed/demo/<clip_id>_positions.parquet, a side-by-side mp4 and 2 stills.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pipeline.field_mapping.field import HASH_FROM_SIDELINE_BY_LEVEL, WIDTH
from pipeline.field_mapping.homography import camera_mismatch, fit_frame, to_field
from pipeline.field_mapping.lines import grass_mask
from scripts.detect_field_lines import LEVEL, draw_grid

OUT = Path("data/processed/demo")
PX = 8  # top-down panel scale, px per yard


def frame_motion(prev_gray, gray, mask):
    """Homography taking pixels of the previous frame to this one, from grass texture."""
    pts = cv2.goodFeaturesToTrack(prev_gray, 400, 0.01, 8, mask=mask)
    if pts is None or len(pts) < 8:
        return None
    nxt, ok, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
    ok = ok.ravel() == 1
    if ok.sum() < 8:
        return None
    F, _ = cv2.findHomography(pts[ok], nxt[ok], cv2.RANSAC, 2.0)  # RANSAC drops moving players
    return F


def analyse(path):
    """Per frame: own fit (H, rms, mismatch) or None, and motion F from the previous frame."""
    cap = cv2.VideoCapture(str(path))
    fits, motion, prev = [], [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H, rms, _ = fit_frame(frame, LEVEL)
        fits.append(None if H is None else (H, rms, camera_mismatch(H, frame.shape[1::-1])))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion.append(None if prev is None else frame_motion(prev, gray, grass_mask(frame)))
        prev = gray
    cap.release()
    return fits, motion


def chain(fits, motion):
    """Carry the anchor's H to all frames. H maps image -> field, F maps prev px -> this px."""
    good = [i for i, f in enumerate(fits) if f is not None]
    if not good:
        return None, None
    a = min(good, key=lambda i: (fits[i][2], fits[i][1]))  # most plausible camera, then rms
    Hs = [None] * len(fits)
    Hs[a] = fits[a][0]
    for t in range(a + 1, len(fits)):  # forward: pixel_t -> pixel_{t-1} -> field
        F = motion[t] if motion[t] is not None else np.eye(3)
        Hs[t] = Hs[t - 1] @ np.linalg.inv(F)
    for t in range(a - 1, -1, -1):  # backward: pixel_t -> pixel_{t+1} -> field
        F = motion[t + 1] if motion[t + 1] is not None else np.eye(3)
        Hs[t] = Hs[t + 1] @ F
    return Hs, a


def panel(rows, x_lo, x_hi, h_out):
    """Top-down field image (relative x) with one dot per player."""
    hy = HASH_FROM_SIDELINE_BY_LEVEL[LEVEL]
    w_px, h_px = int((x_hi - x_lo) * PX), int(WIDTH * PX)
    img = np.full((h_px, w_px, 3), (40, 110, 40), np.uint8)
    for x in np.arange(np.ceil(x_lo / 5) * 5, x_hi, 5):
        cv2.line(img, (int((x - x_lo) * PX), 0), (int((x - x_lo) * PX), h_px), (230, 230, 230), 1)
        cv2.putText(img, f"{x:+.0f}", (int((x - x_lo) * PX) + 2, 12), cv2.FONT_HERSHEY_PLAIN, 0.8, (230, 230, 230))
    for y in (hy, WIDTH - hy):
        yy = h_px - int(y * PX)  # y = 0 (camera side) at the bottom, like the video
        cv2.line(img, (0, yy), (w_px, yy), (200, 200, 200), 1, cv2.LINE_4)
    for r in rows.itertuples():
        colour = tuple(int(c) for c in np.random.default_rng(r.track_id).integers(60, 255, 3))
        p = (int((r.x - x_lo) * PX), h_px - int(r.y * PX))
        cv2.circle(img, p, 4, colour, -1)
    return cv2.resize(img, (int(w_px * h_out / h_px), h_out))


def main():
    clip = sys.argv[1] if len(sys.argv) > 1 else "wincam_winchester_2021_q2_snap"
    path = Path("data/raw/clips") / f"{clip}.mp4"
    tracks = pd.read_parquet(f"data/processed/tracks/{clip}.parquet")
    tracks = tracks[tracks.cls == 0]  # people only; the ball is usually in the air, not on the field
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    fits, motion = analyse(path)
    Hs, anchor = chain(fits, motion)
    if Hs is None:
        print(f"{clip}: no mappable frame, nothing to demo")
        return
    print(f"{clip}: {sum(f is not None for f in fits)}/{len(fits)} frames fit on their own; "
          f"anchor frame {anchor} (rms {fits[anchor][1]:.2f} yd, camera mismatch {fits[anchor][2]:.3f}); "
          f"{sum(m is None for m in motion[1:])} frames with no motion estimate")

    feet = np.c_[(tracks.x1 + tracks.x2) / 2, tracks.y2]
    xy = np.full((len(tracks), 2), np.nan)
    for f, idx in tracks.groupby("frame").indices.items():
        if f < len(Hs):
            xy[idx] = to_field(Hs[f], feet[idx])
    pos = pd.DataFrame({
        "clip_id": clip, "frame": tracks.frame.values, "time_s": tracks.frame.values / fps,
        "track_id": tracks.track_id.values, "x": xy[:, 0], "y": xy[:, 1], "conf": tracks.conf.values,
        "h_rms": fits[anchor][1], "x_is_relative": True,
    })
    on_field = pos.y.between(-5, WIDTH + 5)  # feet mapped far off the field = bad box or bad H
    OUT.mkdir(parents=True, exist_ok=True)
    pos.to_parquet(OUT / f"{clip}_positions.parquet", index=False)
    print(f"{len(pos)} positions, {on_field.mean():.0%} land within 5 yd of the field")

    x_lo, x_hi = np.nanpercentile(pos.x[on_field], [2, 98]) + [-5, 5]
    cap = cv2.VideoCapture(str(path))
    writer, stills = None, {len(Hs) // 3, 2 * len(Hs) // 3}
    by_frame = dict(tuple(pos[on_field].groupby("frame")))
    for f in range(len(Hs)):
        ok, frame = cap.read()
        if not ok:
            break
        left = draw_grid(frame, Hs[f])
        for r in tracks[tracks.frame == f].itertuples():
            cv2.rectangle(left, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), (255, 255, 255), 1)
        right = panel(by_frame.get(f, pos.iloc[:0]), x_lo, x_hi, frame.shape[0])
        out = np.hstack([left, right])
        cv2.putText(out, f"frame {f}  x relative (offset unknown)", (10, out.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if writer is None:
            writer = cv2.VideoWriter(str(OUT / f"{clip}_topdown.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     out.shape[1::-1])
        writer.write(out)
        if f in stills:
            cv2.imwrite(str(OUT / f"{clip}_still_{f:05d}.jpg"), out)
    writer.release()
    cap.release()
    print(f"wrote {OUT}/{clip}_topdown.mp4 and 2 stills")


if __name__ == "__main__":
    main()
