import cv2
import numpy as np

from pipeline.field_mapping import field
from pipeline.field_mapping.homography import camera_homography, fit_frame, to_field
from scripts.field_mapping_error_budget import IMAGE, PRESETS, project

PX = 10  # top-down render scale, px per yard


def render(preset, level="hs"):
    """Synthetic camera frame: a painted top-down field warped through a pinhole camera."""
    rng = np.random.default_rng(0)
    top = np.zeros((round(field.WIDTH * PX), round(field.LENGTH * PX), 3), np.uint8)
    top[:] = (40, 130, 45)
    top = cv2.add(top, rng.integers(0, 20, top.shape, dtype=np.uint8))  # turf texture
    white = (235, 235, 235)
    for x in field.YARD_LINES:
        cv2.line(top, (int(x * PX), 0), (int(x * PX), top.shape[0]), white, 2)
    for y in field.SIDELINES:
        cv2.line(top, (0, int(y * PX)), (top.shape[1], int(y * PX)), white, 2)
    hy = field.HASH_FROM_SIDELINE_BY_LEVEL[level]
    for x in range(11, 110):  # a 2-ft dash every yard; on 5-yd lines the line covers it
        for y in (hy, field.WIDTH - hy):
            cv2.line(top, (x * PX, int((y - 0.33) * PX)), (x * PX, int((y + 0.33) * PX)), white, 2)
    H = camera_homography(*PRESETS[preset], IMAGE) @ np.diag([1 / PX, 1 / PX, 1])
    return cv2.warpPerspective(top, H, IMAGE, flags=cv2.INTER_AREA, borderValue=(40, 40, 40))


def test_synthetic_frame_maps_up_to_a_5_yard_offset():
    frame = render("all22")
    H, rms, reason = fit_frame(frame)
    assert reason == "ok", reason
    probe = np.random.default_rng(1).uniform([0, 0], [field.LENGTH, field.WIDTH], (2000, 2))
    pix, vis = project(camera_homography(*PRESETS["all22"], IMAGE), probe)
    got, true = to_field(H, pix[vis]), probe[vis]
    offset = np.median(got[:, 0] - true[:, 0])
    # x is relative: the found lines are right, but which yard line is which is unknown
    assert abs(offset - 5 * round(offset / 5)) < 0.25
    err = np.linalg.norm(got - true - [offset, 0], axis=1)
    # Painted lines are ~2 px wide and get resampled by the warp, so each landmark is found
    # to ~1-2 px. The error budget gives p95 ~0.1 yd at 2 px for this camera; 0.5 yd allows
    # for the far end (1 px ~ 0.1 yd there) and stays under a player's footprint.
    assert np.percentile(err, 95) < 0.5
