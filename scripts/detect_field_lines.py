"""Per-clip field-mapping report on real clips: mappable rate, rms, failure reasons.

Samples every Nth frame of each clip in data/raw/clips/, fits a homography per frame, and
saves QA overlays (first few mapped frames + one failure) (fitted field grid drawn back onto the frame) to
data/processed/field/<clip_id>_<frame>.jpg. Run from repo root:
    uv run python -m scripts.detect_field_lines [N]
"""

import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from pipeline.field_mapping.field import HASH_FROM_SIDELINE_BY_LEVEL, WIDTH
from pipeline.field_mapping.homography import fit_frame

CLIPS = Path("data/raw/clips")
OUT = Path("data/processed/field")
LEVEL = "hs"  # all current clips are US high school (hashes split the field in thirds)
OVERLAYS_PER_CLIP = 3


def draw_grid(frame, H, level=LEVEL):
    """Draw the fitted field (relative yard lines every 5 yd, sidelines, hashes) onto the frame."""
    Hinv = np.linalg.inv(H)  # field -> image
    hy = HASH_FROM_SIDELINE_BY_LEVEL[level]
    out = frame.copy()

    def poly(pts, colour):
        q = np.c_[pts, np.ones(len(pts))] @ Hinv.T
        q = q[q[:, 2] > 0]
        if len(q) > 1:
            cv2.polylines(out, [np.int32(q[:, :2] / q[:, 2:])], False, colour, 2)

    xs = np.arange(-60, 61, 5.0)
    for x in xs:
        poly(np.c_[np.full(20, x), np.linspace(0, WIDTH, 20)], (0, 0, 255))
    for y, colour in ((0, (255, 255, 0)), (WIDTH, (255, 255, 0)), (hy, (0, 255, 255)), (WIDTH - hy, (0, 255, 255))):
        poly(np.c_[np.linspace(xs[0], xs[-1], 50), np.full(50, y)], colour)
    return out


def run(path, step):
    cap = cv2.VideoCapture(str(path))
    reasons, rms, i = Counter(), [], 0
    saved = Counter()  # overlays saved so far: mapped vs failed
    while True:
        ok = cap.grab()  # stream: decode only the frames we sample
        if not ok:
            break
        if i % step == 0:
            _, frame = cap.retrieve()
            H, err, reason = fit_frame(frame, LEVEL)
            reasons[reason] += 1
            if H is not None:
                rms.append(err)
            kind = "ok" if H is not None else "fail"
            if saved[kind] < (OVERLAYS_PER_CLIP if kind == "ok" else 1):
                img = draw_grid(frame, H) if H is not None else frame.copy()
                cv2.putText(img, f"{i} {reason}" + (f" rms {err:.2f} yd" if H is not None else ""),
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imwrite(str(OUT / f"{path.stem}_{i:05d}.jpg"), img)
                saved[kind] += 1
        i += 1
    cap.release()
    return reasons, rms


def main():
    step = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    OUT.mkdir(parents=True, exist_ok=True)
    angles = {c["id"]: c["angle"] for c in json.loads(Path("docs/clips_manifest.json").read_text())}
    print(f"{'clip':<36}{'angle':<19}{'frames':>7}{'mappable':>10}{'med rms yd':>12}  failure reasons")
    for path in sorted(p for p in CLIPS.glob("*.mp4") if not p.name.startswith(".")):
        reasons, rms = run(path, step)
        n = sum(reasons.values())
        fails = ", ".join(f"{r} {c}" for r, c in reasons.most_common() if r != "ok")
        med = f"{np.median(rms):.2f}" if rms else "-"
        print(f"{path.stem:<36}{angles.get(path.stem, '?'):<19}{n:>7}{reasons['ok'] / max(n, 1):>10.0%}{med:>12}  {fails}")


if __name__ == "__main__":
    main()
