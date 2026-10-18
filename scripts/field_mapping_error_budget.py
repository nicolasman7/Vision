"""How far off (in yards) is the image->field map when landmark pixels are noisy?

Synthetic pinhole cameras only, no footage. Run from repo root:
    uv run python -m scripts.field_mapping_error_budget
"""

import cv2
import numpy as np

from pipeline.field_mapping.field import LENGTH, WIDTH, YARD_M, landmarks
from pipeline.field_mapping.homography import camera_homography, fit, to_field

IMAGE = (1920, 1080)
SIGMAS = (0.5, 1.0, 2.0, 4.0)
TRIALS = 200
EVAL_PER_TRIAL = 100
Y = YARD_M  # presets are in metres; field positions written as yards * Y

# (camera position m, look-at point m, horizontal field of view deg). Approximate, not
# measured from any real stadium. Field metres: x 0-109.7 along, y 0-48.8 across.
PRESETS = {
    # All-22 "coaches' film": press-box roof at midfield, 40 m behind the near sideline,
    # 45 m up, wide lens. Sees both sidelines and ~60 yd of length (x 3-81).
    "all22": ((50 * Y, -40, 45), (45 * Y, 24 * Y, 0), 40),
    # Main TV camera: 50-yd line, 45 m back, 25 m up, zoomed so ~28 yd of the near hash
    # is in frame (x 29-58). Sees the near sideline, not the far one.
    "broadcast": ((50 * Y, -45, 25), (45 * Y, 11 * Y, 0), 21),
    # End-zone camera: 15 m behind the end line, centred, 30 m up, looking up the field.
    # Sees full width from the goal line to ~x 64; depth is foreshortened.
    "endzone": ((-15, 26.67 * Y, 30), (30 * Y, 26.67 * Y, 0), 45),
}


def project(H, field_pts):
    """Field yd -> pixels, plus a mask of points in front of the camera and inside the frame."""
    q = np.c_[field_pts, np.ones(len(field_pts))] @ H.T
    uv = q[:, :2] / q[:, 2:]
    ok = (q[:, 2] > 0) & (uv >= 0).all(axis=1) & (uv[:, 0] < IMAGE[0]) & (uv[:, 1] < IMAGE[1])
    return uv, ok


def errors(preset, sigma, rng, rows=None):
    """Field errors (yd) at random visible points over TRIALS noisy fits; None if unmappable.

    rows: keep only landmarks on these rows (e.g. {"sideline_lo"}) to simulate missing hashes.
    """
    H_true = camera_homography(*PRESETS[preset], IMAGE)
    marks = {k: v for k, v in landmarks().items() if rows is None or k.split("_", 1)[1] in rows}
    field_pts = np.array(list(marks.values()))
    pix, ok = project(H_true, field_pts)
    field_pts, pix = field_pts[ok], pix[ok]
    out = []
    for _ in range(TRIALS):
        result = fit(pix + rng.normal(0, sigma, pix.shape), field_pts)
        if result is None:
            return None
        # A fresh set of random field points in view, e.g. players' feet. Their pixels are
        # exact, so the error measured is the homography's alone, not the detector's.
        cand = rng.uniform([0, 0], [LENGTH, WIDTH], (20 * EVAL_PER_TRIAL, 2))
        cand_pix, vis = project(H_true, cand)
        truth, cand_pix = cand[vis][:EVAL_PER_TRIAL], cand_pix[vis][:EVAL_PER_TRIAL]
        out.append(np.linalg.norm(to_field(result[0], cand_pix) - truth, axis=1))
    return np.concatenate(out)


def main():
    cv2.setRNGSeed(0)  # RANSAC sampling
    rng = np.random.default_rng(0)
    print(f"{'view':<28}{'sigma px':>9}{'median yd':>11}{'p95 yd':>9}")
    cases = [(p, None) for p in PRESETS] + [("broadcast", {"sideline_lo"})]
    for preset, rows in cases:
        label = preset if rows is None else f"{preset}, sideline only"
        for sigma in SIGMAS:
            e = errors(preset, sigma, rng, rows)
            stats = f"{'unmappable':>20}" if e is None else f"{np.median(e):>11.3f}{np.percentile(e, 95):>9.3f}"
            print(f"{label:<28}{sigma:>9}{stats}")


if __name__ == "__main__":
    main()
