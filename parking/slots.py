"""Slot geometry: cut a parking slot out of a lot image, and read the slot
annotations of PKLot (XML) and CNR-EXT (per-camera CSV).

A slot is a polygon in pixel coordinates (usually 4 points). It is cropped by
fitting the minimum-area rotated rectangle, rotating that rectangle by the
smallest angle that makes it axis-aligned (so "up" stays roughly "up" in the
image) and warping it to a ``PATCH_SIZE`` square. For an axis-aligned box this
is the same as cropping the box and resizing it, which is how the CNR-EXT
patches were made.
"""

from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from . import PATCH_SIZE


def slot_quad(polygon: np.ndarray) -> np.ndarray:
    """Corners (tl, tr, br, bl) of the polygon's min-area rectangle, float32 (4, 2)."""
    pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    # OpenCV returns angle in (0, 90]; bring it to [-45, 45) so the patch
    # keeps the image's orientation instead of the rectangle's long axis.
    if angle >= 45:
        angle -= 90
        w, h = h, w
    box = cv2.boxPoints(((cx, cy), (max(w, 1.0), max(h, 1.0)), angle))
    s, d = box.sum(1), box[:, 1] - box[:, 0]
    return np.stack([box[s.argmin()], box[d.argmin()], box[s.argmax()], box[d.argmax()]]).astype(np.float32)


def crop_slot(image: np.ndarray, polygon: np.ndarray, size: int = PATCH_SIZE) -> np.ndarray:
    """Warp one slot of an image (H, W, 3) to a (size, size, 3) patch."""
    dst = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    m = cv2.getPerspectiveTransform(slot_quad(polygon), dst)
    return cv2.warpPerspective(image, m, (size, size), flags=cv2.INTER_AREA,
                               borderMode=cv2.BORDER_REPLICATE)


def crop_slots(image: np.ndarray, polygons: list[np.ndarray], size: int = PATCH_SIZE) -> np.ndarray:
    if not polygons:
        return np.zeros((0, size, size, 3), dtype=np.uint8)
    return np.stack([crop_slot(image, p, size) for p in polygons])


def read_pklot_xml(path: Path) -> list[dict]:
    """Slots of one PKLot image: ``[{"id", "polygon", "occupied"}]``.

    ``occupied`` is ``None`` for the few spaces that have no label.
    """
    root = ET.parse(path).getroot()
    slots = []
    for space in root.iter("space"):
        pts = [(float(p.get("x")), float(p.get("y"))) for p in space.iter("point")]
        if len(pts) < 3:  # fall back to the rotated rectangle
            rr = space.find("rotatedRect")
            if rr is None:
                continue
            c, s, a = rr.find("center"), rr.find("size"), rr.find("angle")
            pts = cv2.boxPoints(((float(c.get("x")), float(c.get("y"))),
                                 (float(s.get("w")), float(s.get("h"))), float(a.get("d")))).tolist()
        occ = space.get("occupied")
        slots.append({
            "id": int(space.get("id")),
            "polygon": np.asarray(pts, dtype=np.float32),
            "occupied": None if occ in (None, "") else int(occ),
        })
    return slots


def read_cnr_camera_csv(path: Path, scale: float = 1000 / 2592) -> dict[int, np.ndarray]:
    """Slot boxes of one CNR-EXT camera: ``{slot_id: polygon}``.

    The CSV boxes are in 2592x1944 pixels; ``scale`` maps them onto the
    1000x750 full frames that are distributed.
    """
    slots = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            x, y, w, h = (float(row[k]) * scale for k in ("X", "Y", "W", "H"))
            slots[int(row["SlotId"])] = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    return slots
