"""Cut the openly licensed clips in docs/clips_manifest.json into data/raw/clips/<id>.mp4.

Run from repo root: uv run python -m scripts.fetch_clips
Skips clips that already exist, so re-running only fills gaps. Video only (OpenCV has no audio).
Internet Archive files are streamed and seeked over HTTPS (no full-game download).
Wikimedia files are short; they are fetched once to a temp file because Wikimedia rate-limits
(HTTP 429) the many range requests a seek makes, and its webm files don't seek over HTTP.
"""

import json
import os
import shutil
import tempfile
import urllib.request

import cv2

MANIFEST = "docs/clips_manifest.json"
OUT = "data/raw/clips"
UA = "VisionFilmScout/0.1 (personal research; openly licensed clips only)"


def local_copy(url, tmp):
    path = os.path.join(tmp, "src" + os.path.splitext(url.split("?")[0])[1])
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
        shutil.copyfileobj(r, f)
    return path


def cut(src, dst, start_s, end_s):
    """Write frames in [start_s, end_s) of src to dst. Returns (seconds, w, h, fps)."""
    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if start_s > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    writer, n = None, 0
    # Write in a hidden sibling folder (OpenCV needs the .mp4 extension to pick the container),
    # so readers globbing data/raw/clips/*.mp4 never see a half-written file.
    tmp_dir = os.path.join(os.path.dirname(dst), ".tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_dst = os.path.join(tmp_dir, os.path.basename(dst))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if t < start_s:  # seek landed on an earlier keyframe
            continue
        if t >= end_s:
            break
        if writer is None:
            h, w = frame.shape[:2]
            for codec in ("avc1", "mp4v"):  # H.264 is ~3x smaller; mp4v if this build lacks it
                writer = cv2.VideoWriter(tmp_dst, cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*codec), fps, (w, h))
                if writer.isOpened():
                    break
        writer.write(frame)
        n += 1
    cap.release()
    if writer is None:
        raise RuntimeError(f"no frames between {start_s}s and {end_s}s")
    writer.release()
    check = cv2.VideoCapture(tmp_dst)
    got = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
    check.release()
    if abs(got - n) > 1:
        raise RuntimeError(f"wrote {n} frames but file has {got}")
    os.replace(tmp_dst, dst)  # atomic on the same disk: readers see the old file or the finished one
    return n / fps, w, h, fps


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for c in json.load(open(MANIFEST)):
        dst = os.path.join(OUT, c["id"] + ".mp4")
        status = "exists"
        try:
            if not os.path.exists(dst):
                with tempfile.TemporaryDirectory() as tmp:
                    src = c["file_url"]
                    if "upload.wikimedia.org" in src:
                        src = local_copy(src, tmp)
                    cut(src, dst, c["start_s"], c["end_s"])
                status = "written"
            cap = cv2.VideoCapture(dst)
            fps = cap.get(cv2.CAP_PROP_FPS)
            secs = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
            res = f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
        except Exception as e:  # one bad source (e.g. HTTP 429) shouldn't stop the rest; re-run later
            secs, res, status = 0.0, "-", f"FAILED: {e}"
        rows.append((c["id"], c["angle"], f"{secs:.1f}", res, c["license"], status))

    head = ("id", "angle", "seconds", "resolution", "license", "status")
    widths = [max(len(str(r[i])) for r in rows + [head]) for i in range(len(head))]
    for r in [head] + rows:
        print("  ".join(str(v).ljust(w) for v, w in zip(r, widths)))
    mb = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT) if f.endswith(".mp4")) / 1e6
    print(f"\n{sum(r[5] != 'exists' and not r[5].startswith('FAILED') for r in rows)} written, "
          f"{sum(r[5].startswith('FAILED') for r in rows)} failed, {mb:.0f} MB in {OUT}/")


if __name__ == "__main__":
    main()
