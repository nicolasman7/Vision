import cv2
import numpy as np

from pipeline.field_mapping import field
from pipeline.field_mapping.homography import camera_homography, fit, to_field
from scripts.field_mapping_error_budget import IMAGE, PRESETS, errors, project


def visible_landmarks(preset):
    H = camera_homography(*PRESETS[preset], IMAGE)
    pts = np.array(list(field.landmarks().values()))
    pix, ok = project(H, pts)
    return H, pix[ok], pts[ok]


def test_zero_noise_recovers_h_exactly():
    for preset in PRESETS:
        H_true, pix, pts = visible_landmarks(preset)
        H, rms = fit(pix, pts)
        probe = np.random.default_rng(1).uniform([0, 0], [field.LENGTH, field.WIDTH], (500, 2))
        probe_pix, vis = project(H_true, probe)
        assert rms < 1e-6
        assert np.abs(to_field(H, probe_pix[vis]) - probe[vis]).max() < 1e-6


def test_degenerate_inputs_are_unmappable():
    _, pix, pts = visible_landmarks("broadcast")
    assert fit(pix[:3], pts[:3]) is None  # fewer than 4 points
    on_sideline = pts[:, 1] == 0  # yard lines meeting the near sideline, no hash
    assert on_sideline.sum() >= 4 and fit(pix[on_sideline], pts[on_sideline]) is None
    one_line = pts[:, 0] == pts[0, 0]  # a single yard line, no second x
    assert fit(pix[one_line], pts[one_line]) is None
    # all but one point on the sideline: still one degree of freedom short
    keep = on_sideline | (np.arange(len(pts)) == np.flatnonzero(~on_sideline)[0])
    assert fit(pix[keep], pts[keep]) is None


def test_broadcast_one_pixel_noise_error_bound():
    # Broadcast frames ~28 yd across 1920 px, ~0.015 yd/px at the near hash and a few
    # times that at the far edge. 16 landmarks average the noise down, so with sigma = 1 px
    # the map should be good to a few px-equivalents (measured p95 ~0.04 yd). 0.15 yd
    # (5 in) leaves ~4x headroom and is far below a player's footprint (~0.5 yd).
    cv2.setRNGSeed(0)
    e = errors("broadcast", 1.0, np.random.default_rng(0))
    assert np.percentile(e, 95) < 0.15
