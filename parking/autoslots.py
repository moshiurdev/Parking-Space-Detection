"""Find parking slots automatically from the painted stall markings.

The classifier needs to know where the slots are. In the datasets they are
hand-annotated polygons; for a new image this module proposes them from the
line markings instead, so a user can upload a photo and get a result without
drawing anything.

It is a classical computer-vision heuristic, not a learned detector, and it
assumes the lot is seen from above (drone or high pole) with visible stall
separator lines:

1. thin bright lines are isolated with a morphological top-hat;
2. line segments are found with the probabilistic Hough transform;
3. for each of the two dominant line directions, segments along it are merged
   into stall separators, cut where a perpendicular line (the bay's centre or
   end line) crosses them, and grouped into bays of separators with the same
   extent;
4. inside a bay, close double lines (hairpin markings) are merged, the stall
   width is the median spacing and every gap of 1-3 stall widths becomes
   1-3 slots; one slot is added past each end of the bay when it fits in the
   image.

It fails on oblique views where the lines converge strongly, on lots without
painted separators and when most lines are hidden by cars. The web app then
asks the user to draw the slots by hand.
"""

from __future__ import annotations

import cv2
import numpy as np

WORK_SIZE = 1000  # the longer image side is resized to this before detection
ANGLE_TOL = 10.0  # degrees: segments within this of a direction belong to it


def marking_mask(bgr: np.ndarray) -> np.ndarray:
    """Binary mask of paint markings: thin, elongated, white or yellow
    structures brighter than their surroundings."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    size = max(gray.shape)
    k = max(9, int(round(min(gray.shape) * 0.03)) | 1)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    thr = max(35.0, float(np.percentile(tophat, 99)) * 0.35)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[..., 0], hsv[..., 1]
    paint = (sat < 70) | ((hue >= 12) & (hue <= 40))  # white/grey or yellow, not plants or car paint
    mask = ((tophat > thr) & paint).astype(np.uint8)

    # keep components that are thin (line-like) and long; car bodies and
    # windscreen reflections are either thick or short
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    keep = np.zeros(n, bool)
    max_half_width = max(2.5, size * 0.006)
    min_extent = size * 0.035
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if max(w, h) < min_extent:
            continue
        d = dist[labels == i]
        keep[i] = np.percentile(d, 90) <= max_half_width
    return (keep[labels] * 255).astype(np.uint8)


def find_segments(mask: np.ndarray) -> np.ndarray:
    """Line segments ``(N, 4)`` as x1, y1, x2, y2."""
    size = max(mask.shape)
    lines = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=30,
                            minLineLength=max(15, size * 0.025), maxLineGap=max(3, size * 0.006))
    return np.zeros((0, 4), np.float32) if lines is None else lines.reshape(-1, 4).astype(np.float32)


def painted_on_ground(gray: np.ndarray, segs: np.ndarray, offset: float, contrast: float = 25.0) -> np.ndarray:
    """True for segments that are brighter than the ground on *both* sides.

    A painted line has asphalt on either side; the bright outline of a white
    car has the car body on one side."""
    if not len(segs):
        return np.zeros(0, bool)
    h, w = gray.shape
    t = np.linspace(0.1, 0.9, 9)
    p = segs[:, None, :2] + (segs[:, None, 2:] - segs[:, None, :2]) * t[None, :, None]  # (N, 9, 2)
    d = segs[:, 2:] - segs[:, :2]
    n = np.stack([-d[:, 1], d[:, 0]], 1) / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6)[:, None]

    def sample(pts):
        x = np.clip(np.round(pts[..., 0]).astype(int), 0, w - 1)
        y = np.clip(np.round(pts[..., 1]).astype(int), 0, h - 1)
        return gray[y, x].astype(np.float32)

    line = sample(p)
    side_a = sample(p + n[:, None] * offset)
    side_b = sample(p - n[:, None] * offset)
    return (np.median(line - side_a, 1) > contrast) & (np.median(line - side_b, 1) > contrast)


def marking_segments(bgr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Line segments of the paint markings in a (work-size) image."""
    segs = find_segments(marking_mask(bgr) if mask is None else mask)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return segs[painted_on_ground(gray, segs, offset=max(4.0, max(gray.shape) * 0.008))]


def vegetation_mask(bgr: np.ndarray) -> np.ndarray:
    """Green plants (planters, verges): never inside a parking slot."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return (((hsv[..., 0] >= 35) & (hsv[..., 0] <= 85) & (hsv[..., 1] > 60)) * 255).astype(np.uint8)


def _mask_fraction(quad: np.ndarray, mask: np.ndarray) -> float:
    """Share of mask pixels in the middle of a slot (its inner 60 %)."""
    centre = quad.mean(0)
    inner = (centre + (quad - centre) * 0.6).astype(np.int32)
    region = np.zeros_like(mask)
    cv2.fillPoly(region, [inner], 255)
    n = cv2.countNonZero(region)
    return cv2.countNonZero(mask & region) / n if n else 1.0


def dominant_angles(segs: np.ndarray) -> list[float]:
    """The strongest line direction and the strongest one roughly perpendicular to it (degrees)."""
    if not len(segs):
        return []
    d = segs[:, 2:] - segs[:, :2]
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180
    hist = np.bincount((ang // 2).astype(int) % 90, weights=np.hypot(d[:, 0], d[:, 1]), minlength=90)
    hist = np.convolve(np.r_[hist[-2:], hist, hist[:2]], np.ones(5) / 5, mode="valid")
    a1 = int(hist.argmax())
    window = [(a1 + 45 + o) % 90 for o in range(-8, 9)]  # 90 +- 16 degrees
    a2 = window[int(np.argmax(hist[window]))]
    out = [a1 * 2.0 + 1]
    if hist[a2] > 0:
        out.append(a2 * 2.0 + 1)
    return out


def _to_frame(segs: np.ndarray, theta: float) -> np.ndarray:
    """Rotate segments so direction ``theta`` becomes the u (horizontal) axis."""
    t = np.radians(theta)
    d, n = np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])
    p, q = segs[:, :2], segs[:, 2:]
    u1, v1, u2, v2 = p @ d, p @ n, q @ d, q @ n
    swap = u1 > u2
    u1[swap], u2[swap] = u2[swap], u1[swap].copy()
    v1[swap], v2[swap] = v2[swap], v1[swap].copy()
    return np.stack([u1, v1, u2, v2], 1)


def _from_frame(uv: np.ndarray, theta: float) -> np.ndarray:
    t = np.radians(theta)
    d, n = np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])
    return uv[..., :1] * d + uv[..., 1:] * n


def _angle_of(segs: np.ndarray) -> np.ndarray:
    d = segs[:, 2:] - segs[:, :2]
    return np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180


def _near(a: np.ndarray, theta: float) -> np.ndarray:
    diff = np.abs((a - theta + 90) % 180 - 90)
    return diff <= ANGLE_TOL


def _union_find(n: int, pairs) -> np.ndarray:
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    return np.array([find(i) for i in range(n)])


def merge_collinear(frame_segs: np.ndarray, v_tol: float, gap: float) -> np.ndarray:
    """Merge horizontal segments (u1, v1, u2, v2) lying on one line into lines
    ``(u0, u1, v_at_u0, v_at_u1)`` fitted through their endpoints."""
    if not len(frame_segs):
        return np.zeros((0, 4))
    vm = (frame_segs[:, 1] + frame_segs[:, 3]) / 2
    order = np.argsort(vm)
    pairs = []
    for a in range(len(order)):
        i = order[a]
        for b in range(a + 1, len(order)):
            j = order[b]
            if vm[j] - vm[i] > v_tol:
                break
            if frame_segs[j, 0] <= frame_segs[i, 2] + gap and frame_segs[i, 0] <= frame_segs[j, 2] + gap:
                pairs.append((i, j))
    groups = _union_find(len(frame_segs), pairs)
    lines = []
    for g in np.unique(groups):
        s = frame_segs[groups == g]
        u = np.r_[s[:, 0], s[:, 2]]
        v = np.r_[s[:, 1], s[:, 3]]
        u0, u1 = u.min(), u.max()
        slope = np.polyfit(u, v, 1)[0] if u1 - u0 > 1 and len(u) > 2 else 0.0
        vc = v.mean() - slope * u.mean()
        lines.append((u0, u1, vc + slope * u0, vc + slope * u1))
    return np.array(lines)


def _cut_at_dividers(seps: np.ndarray, divs: np.ndarray, tol: float, min_len: float) -> np.ndarray:
    """Split separators where a perpendicular line crosses them.

    ``divs`` are perpendicular lines in the same frame, given as
    ``(v0, v1, u_at_v0, u_at_v1)`` (they run along v)."""
    out = []
    for u0, u1, va, vb in seps:
        vmid = (va + vb) / 2
        cuts = []
        for v0, v1, ua, ub in divs:
            if not (v0 - tol <= vmid <= v1 + tol):
                continue
            uc = ua + (ub - ua) * (vmid - v0) / max(v1 - v0, 1e-6)
            if u0 + min_len < uc < u1 - min_len:
                cuts.append(uc)
        edges = [u0, *sorted(cuts), u1]
        for a, b in zip(edges[:-1], edges[1:]):
            if b - a >= min_len:
                slope = (vb - va) / max(u1 - u0, 1e-6)
                out.append((a, b, va + slope * (a - u0), va + slope * (b - u0)))
    return np.array(out) if out else np.zeros((0, 4))


def _group_bays(pieces: np.ndarray, rel_tol: float = 0.2) -> list[np.ndarray]:
    """Pieces with nearly the same start and end belong to one bay (one row of stalls)."""
    n = len(pieces)
    length = pieces[:, 1] - pieces[:, 0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            tol = rel_tol * max(length[i], length[j])
            if abs(pieces[i, 0] - pieces[j, 0]) < tol and abs(pieces[i, 1] - pieces[j, 1]) < tol:
                pairs.append((i, j))
    groups = _union_find(n, pairs)
    return [pieces[groups == g] for g in np.unique(groups)]


def _bay_slots(bay: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    """Slot quads (in frame coordinates) of one bay of separator pieces, each
    with a flag that says whether it was extrapolated past the bay's end."""
    if len(bay) < 2:
        return []
    U0, U1 = float(np.median(bay[:, 0])), float(np.median(bay[:, 1]))
    length = U1 - U0
    slopes = (bay[:, 3] - bay[:, 2]) / np.maximum(bay[:, 1] - bay[:, 0], 1e-6)
    v_mid = bay[:, 2] + slopes * ((U0 + U1) / 2 - bay[:, 0])
    order = np.argsort(v_mid)
    v_mid, slopes = v_mid[order], slopes[order]

    # merge double lines (hairpin markings): gaps much smaller than the typical gap
    gaps = np.diff(v_mid)
    ref = np.percentile(gaps, 75)
    groups = [[0]]
    for i, g in enumerate(gaps, start=1):
        if g < 0.45 * ref:
            groups[-1].append(i)
        else:
            groups.append([i])
    v_sep = np.array([v_mid[g].mean() for g in groups])
    s_sep = np.array([slopes[g].mean() for g in groups])
    if len(v_sep) < 2:
        return []
    gaps = np.diff(v_sep)
    pitch = float(np.median(gaps))
    pitch = float(np.median(gaps[gaps < 1.5 * pitch]))
    if not 0.9 <= length / pitch <= 4.5:
        return []

    def edge(v, s):  # a separator as its two end points (u, v) at the bay's ends
        half = length / 2
        return np.array([[U0, v - s * half], [U1, v + s * half]])

    s_mean = float(np.mean(s_sep))
    lines = ([edge(v_sep[0] - pitch, s_mean)] + [edge(v, s) for v, s in zip(v_sep, s_sep)]
             + [edge(v_sep[-1] + pitch, s_mean)])
    quads = []
    for i in range(len(lines) - 1):
        a, b = lines[i], lines[i + 1]
        g = float(np.mean(b[:, 1] - a[:, 1]))
        k = int(round(g / pitch))
        if k < 1 or k > 3 or abs(g / pitch - k) > 0.3 * k:
            continue  # not a whole number of stalls: an aisle or a missed line
        extra = i == 0 or i == len(lines) - 2
        for j in range(k):
            top, bot = a + (b - a) * (j / k), a + (b - a) * ((j + 1) / k)
            quads.append((np.array([top[0], top[1], bot[1], bot[0]]), extra))
    return quads


def _bays_for_direction(segs: np.ndarray, theta: float, size: float) -> list[list[tuple[np.ndarray, bool]]]:
    """Slots, grouped by bay, whose separators run in direction ``theta``."""
    ang = _angle_of(segs)
    along = segs[_near(ang, theta)]
    across = segs[_near(ang, theta + 90)]
    if len(along) < 2:
        return []
    v_tol, gap, min_len = size * 0.006, size * 0.03, size * 0.035
    seps = merge_collinear(_to_frame(along, theta), v_tol, gap)
    seps = seps[seps[:, 1] - seps[:, 0] >= min_len]
    # perpendicular lines (bay centre / end lines), merged in their own frame
    divs = merge_collinear(_to_frame(across, theta + 90), v_tol, gap * 2)
    divs = divs[divs[:, 1] - divs[:, 0] >= min_len]
    # rotating the frame by +90 degrees maps (u', v') to (v, -u)
    divs = np.stack([divs[:, 0], divs[:, 1], -divs[:, 2], -divs[:, 3]], 1) if len(divs) else divs
    pieces = _cut_at_dividers(seps, divs, tol=size * 0.02, min_len=min_len)
    if not len(pieces):
        return []
    return [[(_from_frame(q, theta), extra) for q, extra in _bay_slots(bay)] for bay in _group_bays(pieces)]


def _clip_to_image(quad: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.clip(quad, [0, 0], [w, h]).astype(np.float32)


def _inside_fraction(quad: np.ndarray, w: int, h: int) -> float:
    area = cv2.contourArea(quad.astype(np.float32))
    return cv2.contourArea(_clip_to_image(quad, w, h)) / area if area > 0 else 0.0


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    inter, _ = cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))
    return inter / max(1e-6, min(cv2.contourArea(a.astype(np.float32)), cv2.contourArea(b.astype(np.float32))))


def detect_slots(image_bgr: np.ndarray) -> list[np.ndarray]:
    """Slot polygons ``(4, 2)`` in the input image's pixel coordinates."""
    h, w = image_bgr.shape[:2]
    scale = WORK_SIZE / max(h, w)
    small = cv2.resize(image_bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    segs = marking_segments(small)
    plants = vegetation_mask(small)
    bays = []
    for theta in dominant_angles(segs):
        for bay in _bays_for_direction(segs, theta, max(sh, sw)):
            # slots may be cut by the image border; a slot extrapolated past the
            # bay's last line is dropped when it lands on plants (a planter)
            bay = [q for q, extra in bay if _inside_fraction(q, sw, sh) >= 0.7
                   and not (extra and _mask_fraction(q, plants) > 0.15)]
            if len(bay) >= 2:
                bays.append(bay)
    # bigger bays first; a bay that mostly overlaps accepted slots is a duplicate or noise
    bays.sort(key=len, reverse=True)
    slots: list[np.ndarray] = []
    for bay in bays:
        fresh = [q for q in bay if not any(_overlap(q, s) > 0.3 for s in slots)]
        if len(fresh) >= max(2, len(bay) / 2):
            slots += fresh
    if slots:  # slivers cut off by the image border or by a stray line
        areas = np.array([cv2.contourArea(_clip_to_image(q, sw, sh)) for q in slots])
        slots = [q for q, a in zip(slots, areas) if a >= 0.4 * np.median(areas)]
    return [np.clip(q / scale, [0, 0], [w - 1, h - 1]).astype(np.float32) for q in slots]
