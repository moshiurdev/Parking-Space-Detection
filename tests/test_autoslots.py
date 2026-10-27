"""Automatic slot detection from painted stall lines, on a synthetic top-down lot."""

import cv2
import numpy as np
import pytest

from conftest import marked_lot_image
from parking.autoslots import detect_slots


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter, _ = cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))
    return inter / (cv2.contourArea(a.astype(np.float32)) + cv2.contourArea(b.astype(np.float32)) - inter)


def rotate(img: np.ndarray, polys: list[np.ndarray], degrees: float):
    """Rotate (and shrink a little, so the whole lot stays in the picture)."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 0.75)
    out = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return out, [cv2.transform(p[None], m)[0] for p in polys]


@pytest.mark.parametrize("degrees", [0, 30, 90])
def test_finds_every_stall_between_lines(degrees):
    img, stalls = marked_lot_image()
    img, stalls = rotate(img, stalls, degrees)
    found = detect_slots(img)
    for stall in stalls:
        assert max(iou(stall, f) for f in found) > 0.8
    # plus one stall past the last line of each bay, but none on the planter
    assert len(found) == len(stalls) + 3
    planter = rotate(np.zeros((800, 1000, 3), np.uint8), [np.float32([[110, 15], [340, 15], [340, 85], [110, 85]])],
                     degrees)[1][0]
    assert all(iou(planter, f) < 0.2 for f in found)


def test_works_on_a_smaller_image():
    img, stalls = marked_lot_image()
    found = detect_slots(cv2.resize(img, (500, 400), interpolation=cv2.INTER_AREA))
    for stall in stalls:
        assert max(iou(stall / 2, f) for f in found) > 0.8


@pytest.mark.parametrize("kind", ["blank", "cars only"])
def test_no_lines_no_slots(kind):
    img = np.full((600, 800, 3), 100, np.uint8)
    if kind == "cars only":
        for x in range(50, 700, 150):
            cv2.rectangle(img, (x, 200), (x + 90, 400), (230, 230, 230), -1)
    assert detect_slots(img) == []
