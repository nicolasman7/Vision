"""Find field landmarks (yard line x sideline/hash crossings) in a real frame. Classical OpenCV.

Assumes a sideline-type view: yard lines run "up" the image toward a vanishing point,
sidelines and hash rows run roughly left-right. Which yard line is which needs the painted
numbers, which we don't read, so x is RELATIVE: the leftmost found line is x = 0 and the
true offset is an unknown multiple of 5 yd. y assumes the camera is on the y = 0 sideline.
"""

import cv2
import numpy as np

from pipeline.field_mapping.field import HASH_FROM_SIDELINE_BY_LEVEL, WIDTH


def grass_mask(frame):
    """Largest green region, with holes (players, paint) closed."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (30, 40, 30), (90, 255, 255))
    k = max(3, frame.shape[1] // 40) | 1
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(green)
    if n < 2:
        return np.zeros_like(green)
    return np.uint8(lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])) * 255


def paint_mask(frame, grass):
    """Thin bright unsaturated marks on the grass: brighter than the local background (top-hat)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    k = max(5, frame.shape[1] // 80) | 1
    tophat = cv2.morphologyEx(hsv[..., 2], cv2.MORPH_TOPHAT, np.ones((k, k), np.uint8))
    # Paint is surrounded by grass; jerseys, crowd and graphics mostly are not.
    green = cv2.inRange(hsv, (30, 40, 30), (90, 255, 255))
    near_green = cv2.blur(green, (k, k)) > 100
    return np.uint8((tophat > 35) & (hsv[..., 1] < 70) & near_green & (grass > 0)) * 255


def _line(p, q):
    """Homogeneous line through two image points."""
    return np.cross([*p, 1.0], [*q, 1.0])


def _x_at_row(line, v):
    a, b, c = line
    return -(b * v + c) / a


def _merge(segs, key, tol):
    """Group segments whose key (crossing position) is within tol; fit one line per group."""
    order = np.argsort(key)
    groups, cur = [], [order[0]]
    for i in order[1:]:
        if key[i] - key[cur[-1]] < tol:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    lines = []
    for g in groups:
        pts = segs[g].reshape(-1, 2).astype(np.float32)
        length = np.linalg.norm(segs[g, 2:] - segs[g, :2], axis=1).sum()
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((_line((x0, y0), (x0 + vx, y0 + vy)), length))
    return lines


def yard_lines(paint, grass):
    """Yard lines sorted left to right as homogeneous lines, plus their crossing at a mid row."""
    h, w = paint.shape
    segs = cv2.HoughLinesP(paint, 1, np.pi / 360, 40, minLineLength=h // 12, maxLineGap=w // 60)
    if segs is None:
        return [], None
    segs = segs.reshape(-1, 4).astype(float)
    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])) % 180
    segs = segs[np.abs(ang - 90) < 65]  # steeper than 25 deg: yard-line family
    rows = np.flatnonzero(grass.any(axis=1))
    if len(segs) == 0 or len(rows) == 0:
        return [], None
    v_mid = (rows[0] + rows[-1]) / 2
    key = np.array([_x_at_row(_line(s[:2], s[2:]), v_mid) for s in segs])
    lines = [l for l, length in _merge(segs, key, w / 80) if length > h / 6]
    lines = _keep_concurrent(lines, w)
    lines.sort(key=lambda l: _x_at_row(l, v_mid))
    return lines, v_mid


def _keep_concurrent(lines, w):
    """Drop lines that don't pass near the common vanishing point (logos, other sports' marks)."""
    if len(lines) < 3:
        return lines
    L = np.array([l / np.linalg.norm(l[:2]) for l in lines])
    vp = np.linalg.svd(L)[2][-1]  # point minimising summed distance^2 to all lines
    if abs(vp[2]) < 1e-9:
        return lines  # lines parallel in the image; nothing to test against
    d = np.abs(L @ (vp / vp[2]))
    keep = d < max(w / 50, 3 * np.median(d))
    return [l for l, k in zip(lines, keep) if k]


def _indices(cross, tol):
    """Integer 5-yd index of each sorted crossing, allowing one missed or one spurious line.

    Equally spaced field lines cross any image line at u = (a*i + b) / (c*i + 1), a 1D
    perspective map. Try the simplest story first (consecutive lines), then one missed
    line, then one spurious line; take the first whose fit error is under tol px.
    Returns (idx, keep_mask) or None.
    """
    n = len(cross)
    plain = [(n, None)]
    gaps = [(g, None) for g in range(1, n)]
    drops = [(n, d) for d in range(n)] if n > 4 else []
    for gap, drop in plain + gaps + drops:
        keep = np.arange(n) != drop
        idx = np.cumsum(keep) - 1 + (np.arange(n) >= gap)
        u, i = cross[keep], idx[keep]
        A = np.c_[i, np.ones(len(i)), -i * u]
        sol = np.linalg.lstsq(A, u, rcond=None)[0]
        if len(u) <= 3 or np.abs(A @ sol - u).max() < tol:
            return idx[keep], keep
    return None


def _far_end(line, paint):
    """Top end (u, v) of the longest painted stretch along a yard line, or None if the
    line runs off the top of the frame. Gaps shorter than h/30 (a player standing on the
    line) are bridged."""
    h, w = paint.shape
    v = np.arange(h)
    u = np.round(_x_at_row(line, v)).astype(int)
    ok = (u >= 0) & (u < w)
    on = np.flatnonzero(ok & (paint[v, np.clip(u, 0, w - 1)] > 0))
    if len(on) == 0:
        return None
    runs = np.split(on, np.flatnonzero(np.diff(on) > h // 30) + 1)
    top = max(runs, key=len)[0]
    return (u[top], top) if top > 2 and ok[max(top - 3, 0)] else None


def cross_rows(paint, ylines, w):
    """Lines that cross the yard-line family. Returns [(kind, line, mean_v)] top to bottom.

    - far sideline: yard lines stop there, so it is the line through their top ends.
      ponytail: the near sideline (bottom ends) is skipped; scorebugs and benches hide it.
    - hash rows: short dashes, away from the yard lines, regularly spaced along a line.
    """
    out = []
    thick = cv2.dilate(paint, np.ones((3, 3), np.uint8))
    tops = np.array([t for t in (_far_end(l, thick) for l in ylines) if t is not None], float)
    v_min = 0.0
    if len(tops) >= 3:
        vx, vy, x0, y0 = cv2.fitLine(tops.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        side = _line((x0, y0), (x0 + vx, y0 + vy))
        side = side / np.linalg.norm(side[:2])
        if np.median(np.abs(np.c_[tops, np.ones(len(tops))] @ side)) < w / 150 and abs(vy) < abs(vx):
            out.append(("sideline", side, float(tops[:, 1].mean())))
            v_min = tops[:, 1].max() + w / 100
    n, _, stats, cent = cv2.connectedComponentsWithStats(paint)
    area = stats[1:, cv2.CC_STAT_AREA]
    size = np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT])
    c = cent[1:][(area >= 3) & (size < w / 40) & (cent[1:, 1] > v_min)]
    if len(ylines) and len(c):
        L = np.array([l / np.linalg.norm(l[:2]) for l in ylines])
        c = c[np.abs(np.c_[c, np.ones(len(c))] @ L.T).min(axis=1) > w / 200]
    out += [("hash", l, v) for l, v in _dash_rows(c, w) if _on_painted_lines(l, ylines, thick) >= 0.5]
    out.sort(key=lambda r: r[2])
    return out


def _on_painted_lines(row, ylines, thick):
    """Fraction of yard lines that are painted where this row crosses them. Real hash rows
    cross the field; rows of crowd or jersey blobs off the field don't."""
    h, w = thick.shape
    hits = []
    for l in ylines:
        p = np.cross(l, row)
        u, v = np.round(p[:2] / p[2]).astype(int) if abs(p[2]) > 1e-9 else (-1, -1)
        if 0 <= u < w and 0 <= v < h:
            hits.append(thick[max(v - 2, 0) : v + 3, max(u - 2, 0) : u + 3].any())
    return np.mean(hits) if hits else 0.0


def _regular_run(x):
    """Longest run of sorted x positions with smoothly varying gaps (hash marks: one per yard,
    a doubled gap where a yard line hides one). Crowd and player blobs are irregular."""
    order = np.argsort(x)
    g = np.diff(x[order])
    best, start = (0, 0), 0
    for j in range(1, len(g) + 1):
        r = g[j] / g[j - 1] if j < len(g) and g[j - 1] > 0 else 0
        if not (0.6 < r < 1.6 or 0.4 < r < 0.6 or 1.6 < r < 2.4):
            if j - start > best[1] - best[0]:
                best = (start, j)
            start = j
    return order[best[0] : best[1] + 1]


def _dash_rows(c, w, min_dashes=6, max_rows=3):
    """RANSAC for near-horizontal, regularly spaced rows of dash centroids."""
    rng = np.random.default_rng(0)
    tol = max(2.0, w / 300)  # px; loose enough for slight lens bending of the row
    rows = []
    while len(c) >= min_dashes and len(rows) < max_rows:
        best = None
        for _ in range(500):
            p, q = c[rng.choice(len(c), 2, replace=False)]
            if abs(q[1] - p[1]) > 0.27 * abs(q[0] - p[0]) or abs(q[0] - p[0]) < w / 30:
                continue  # need a near-horizontal pair, some distance apart
            l = _line(p, q)
            inl = np.flatnonzero(np.abs(np.c_[c, np.ones(len(c))] @ (l / np.linalg.norm(l[:2]))) < tol)
            run = inl[_regular_run(c[inl, 0])]
            if best is None or len(run) > len(best):
                best = run
        if best is None or len(best) < min_dashes:
            break
        pts = c[best].astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        rows.append((_line((x0, y0), (x0 + vx, y0 + vy)), float(pts[:, 1].mean())))
        c = np.delete(c, best, axis=0)
    return rows


def _labelings(rows, level):
    """Every plausible way to give the found rows a field y. Image top = far side (camera on
    the y = 0 sideline): far sideline -> y = WIDTH, then far hash, then near hash; a lone
    hash row may be either.

    Spurious rows (crowd, graphics) can't be ruled out here, so each subset in top-to-bottom
    order is a candidate; homography.fit_frame keeps the one a real camera could produce.
    """
    hy = HASH_FROM_SIDELINE_BY_LEVEL[level]
    sides = [[]] + [[(r[1], WIDTH)] for r in rows if r[0] == "sideline"]
    hashes = [r for r in rows if r[0] == "hash"]
    hash_sets = [[(hashes[i][1], WIDTH - hy), (hashes[j][1], hy)] for i in range(len(hashes)) for j in range(i + 1, len(hashes))]
    hash_sets += [[(r[1], y)] for r in hashes for y in (WIDTH - hy, hy)]  # lone row: far or near
    return [s + hs for s in sides for hs in hash_sets + [[]] if len(s + hs) >= 2]


def find_landmarks(frame, level="hs"):
    """Candidate landmark sets for one frame. Returns ([(image_pts, field_pts), ...], reason)."""
    h, w = frame.shape[:2]
    grass = grass_mask(frame)
    if grass.mean() / 255 < 0.15:
        return [], "no_grass"
    paint = paint_mask(frame, grass)
    ylines, v_mid = yard_lines(paint, grass)
    if len(ylines) < 2:
        return [], "<2 yard lines"
    found = _indices(np.array([_x_at_row(l, v_mid) for l in ylines]), w / 150)
    if found is None:
        return [], "yard-line spacing inconsistent"
    idx, keep = found
    ylines = [l for l, k in zip(ylines, keep) if k]
    rows = cross_rows(paint, ylines, w)
    if not rows:
        return [], "no cross-field landmark"
    cands = []
    for labels in _labelings(rows, level):
        img, fld = [], []
        for row, y in labels:
            for yl, i in zip(ylines, idx):
                p = np.cross(yl, row)
                if abs(p[2]) < 1e-9:
                    continue
                u, v = p[:2] / p[2]
                if -0.05 * w <= u <= 1.05 * w and 0 <= v < h:
                    img.append((u, v))
                    fld.append((5.0 * i, y))
        # need 2+ rows, and 6+ points: 4-5 fit (nearly) exactly, hiding a wrong label
        if len({y for _, y in fld}) >= 2 and len(img) >= 6:
            cands.append((np.array(img), np.array(fld)))
    return cands, "ok" if cands else "1 cross-field row or <6 landmarks"
