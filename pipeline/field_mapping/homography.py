"""Image <-> field homography from known landmarks.

The field is flat, so any camera view of it is a plane-to-plane projective map: a 3x3
matrix H (8 degrees of freedom) with [X, Y, 1] ~ H @ [u, v, 1]. Four landmarks in
general position pin it down; more landmarks average out pixel noise.
"""

import cv2
import numpy as np

from pipeline.field_mapping.field import YARD_M

# RANSAC rejects a landmark whose reprojection misses by more than this many pixels.
# A mislabeled landmark (wrong yard line) misses by 5 yd = tens to hundreds of px, while
# honest pixel noise of a few px stays well under it.
RANSAC_PX = 10.0


def _collinear(pts):
    """True if the points lie on one line (second singular value ~ 0)."""
    s = np.linalg.svd(pts - pts.mean(axis=0), compute_uv=False)
    return s[1] < 1e-6 * max(s[0], 1.0)


def fit(image_pts, field_pts):
    """Fit H (image px -> field yd). Returns (H, rms_error_yards) or None if unmappable.

    Unmappable cases:
    - Fewer than 4 points: H has 8 unknowns, each point gives 2 equations.
    - The field points don't span both axes, e.g. yard lines meeting the near sideline
      but no hash: every point has y = 0. Those points only fix how one line of the
      field maps to the image; any stretch or tilt across the field (along y) keeps
      them in place, so a pixel's distance from the sideline is unknowable.
    - All but one point on a single line: a line of points fixes 5 of the 8 degrees of
      freedom no matter how many points are on it, the extra point adds 2, total 7 < 8.
    """
    img = np.asarray(image_pts, dtype=np.float64).reshape(-1, 2)
    fld = np.asarray(field_pts, dtype=np.float64).reshape(-1, 2)
    if len(fld) < 4 or _collinear(fld) or any(_collinear(np.delete(fld, i, 0)) for i in range(len(fld))):
        return None
    # RANSAC field -> image so the outlier threshold is in pixels, where the noise lives.
    H_fi, inliers = cv2.findHomography(fld, img, cv2.RANSAC, RANSAC_PX)
    if H_fi is None:
        return None
    # OpenCV works in float32 internally (~1e-6 yd floor), so refit the inliers in float64.
    keep = inliers.ravel().astype(bool)
    if keep.sum() < 4 or _collinear(fld[keep]):
        return None  # RANSAC kept too few landmarks to trust
    H = _dlt(img[keep], fld[keep])
    rms = float(np.sqrt(np.mean(np.sum((to_field(H, img) - fld) ** 2, axis=1))))
    return H, rms


def _dlt(src, dst):
    """Least-squares H with dst ~ H src (direct linear transform, Hartley-normalised).

    Each pair gives two linear equations in the 9 entries of H; the solution is the
    singular vector with the smallest singular value. Points are first shifted to mean 0
    and scaled to mean distance sqrt(2) so pixel-sized and yard-sized numbers don't
    swamp each other in the SVD.
    """

    def normalise(p):
        c = p.mean(axis=0)
        s = np.sqrt(2) / np.linalg.norm(p - c, axis=1).mean()
        return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]]), (p - c) * s

    Ts, a = normalise(src)
    Td, b = normalise(dst)
    rows = []
    for (x, y), (u, v) in zip(a, b):
        rows += [[-x, -y, -1, 0, 0, 0, u * x, u * y, u], [0, 0, 0, -x, -y, -1, v * x, v * y, v]]
    h = np.linalg.svd(np.array(rows))[2][-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ h @ Ts
    return H / H[2, 2]


def to_field(H, pixels):
    """Map pixel (u, v) points to field (x, y) yards. Pixels above the horizon give nonsense."""
    pts = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def camera_homography(position_m, look_at_m, hfov_deg, image_size):
    """Field (yd) -> image (px) homography of a pinhole camera. For synthetic tests only.

    World frame in metres: x along the field, y across, z up, origin at field (0, 0).
    A ground point (X, Y, 0) projects as K (R [X, Y, 0] + t) = K [r1 r2 t] [X, Y, 1],
    since the z column of R is multiplied by 0. That 3x3 is the homography.
    """
    w, h = image_size
    C = np.asarray(position_m, dtype=np.float64)
    fwd = np.asarray(look_at_m, dtype=np.float64) - C
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])  # horizontal, so the horizon stays level
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)  # image v grows downward
    R = np.stack([right, down, fwd])  # rows: camera axes in world coords
    t = -R @ C
    f = (w / 2) / np.tan(np.radians(hfov_deg) / 2)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    yd_to_m = np.diag([YARD_M, YARD_M, 1.0])
    return K @ np.column_stack([R[:, 0], R[:, 1], t]) @ yd_to_m


# A camera with square pixels and a centred principal point can only produce some
# homographies (see camera_mismatch). Real lenses and zoom add a little; a wrong row label
# (e.g. a crowd row taken for the far hash) stretches the field and adds a lot.
MISMATCH_MAX = 0.25
RMS_MAX = 1.0  # yd; above this the landmarks disagree with each other


def camera_mismatch(H, image_size):
    """How far image->field H is from any pinhole camera with unknown focal length f.

    Field->image is K [r1 r2 t] with r1, r2 orthonormal. After removing the principal
    point, columns h1, h2 give r1 ~ (h1x/f, h1y/f, h1z), r2 likewise. Solve f from either
    "r1 . r2 = 0" or "|r1| = |r2|" (one can be degenerate, e.g. a centred end-zone camera),
    then score the other: 0 = a real camera could take this picture.
    """
    w, h = image_size
    G = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1]]) @ np.linalg.inv(H)
    G = G / np.linalg.norm(G)
    (ax, bx), (ay, by), (az, bz) = G[:, 0:2]
    tries = []
    if abs(az * bz) > 1e-12:
        tries.append(-(ax * bx + ay * by) / (az * bz))
    if abs(bz**2 - az**2) > 1e-12:
        tries.append((ax**2 + ay**2 - bx**2 - by**2) / (bz**2 - az**2))
    best = 1.0
    for f2 in tries:
        if f2 > 0:
            r1, r2 = np.array([ax, ay, az * np.sqrt(f2)]), np.array([bx, by, bz * np.sqrt(f2)])
            n1, n2 = np.linalg.norm(r1), np.linalg.norm(r2)
            best = min(best, max(abs(r1 @ r2) / (n1 * n2), abs(n1 - n2) / (n1 + n2)))
    return float(best)


def fit_frame(frame, level="hs"):
    """Real frame -> (H, rms_yards, reason). H is None unless reason == "ok".

    x in the result is RELATIVE (leftmost found yard line = 0; true offset unknown, a
    multiple of 5 yd). Tries each candidate row labelling from lines.py and keeps the one
    with the most rows, then the most physical camera. See lines.py for the assumptions.
    """
    from pipeline.field_mapping.lines import find_landmarks

    cands, reason = find_landmarks(frame, level)
    if not cands:
        return None, None, reason
    best, reason = None, "degenerate landmarks"
    for img, fld in cands:
        result = fit(img, fld)
        if result is None:
            continue
        H, rms = result
        mis = camera_mismatch(H, frame.shape[1::-1])
        if rms > RMS_MAX or mis > MISMATCH_MAX:
            reason = "implausible camera" if rms <= RMS_MAX else "landmarks disagree"
            continue
        key = (-len(set(fld[:, 1])), mis)
        if best is None or key < best[0]:
            best = (key, H, rms)
    if best is None:
        return None, None, reason
    return best[1], best[2], "ok"
