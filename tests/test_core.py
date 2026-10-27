"""Models, slot cropping, annotation parsing, splits, augmentation and data I/O."""

import numpy as np
import pytest
import torch

from conftest import lot_image
from parking import PATCH_SIZE
from parking.augment import PRESETS, augment
from parking.data import PatchWriter, class_weights, iterate_batches, load_patch_set
from parking.models import build_model, count_parameters
from parking.slots import crop_slot, read_cnr_camera_csv, read_pklot_xml, slot_quad
from parking.splits import split_cnr_days, split_days


@pytest.mark.parametrize("arch,params", [("mobilenet_v3_small", (1.4e6, 1.6e6)), ("malexnet", (5e4, 6e4))])
def test_model_shapes_and_size(arch, params):
    model = build_model(arch, pretrained=False).eval()
    out = model(torch.rand(4, 3, PATCH_SIZE, PATCH_SIZE))
    assert out.shape == (4, 2)
    assert params[0] < count_parameters(model) < params[1]


def test_crop_axis_aligned_slot_is_plain_crop():
    img, slots = lot_image()
    empty, car = crop_slot(img, slots[0]), crop_slot(img, slots[1])
    assert empty.shape == (PATCH_SIZE, PATCH_SIZE, 3)
    assert np.abs(empty.astype(int) - 90).max() <= 1          # all grey asphalt
    assert car[PATCH_SIZE // 2, PATCH_SIZE // 2, 0] > 150      # red car in the middle


def test_slot_quad_keeps_image_orientation():
    # a rectangle rotated by 30 degrees: the patch should be rotated back by 30, not by 120
    c, w, h, a = np.array([100.0, 100.0]), 40, 80, np.deg2rad(30)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]) @ rot.T + c
    tl, tr, br, bl = slot_quad(corners)
    assert np.linalg.norm(tr - tl) == pytest.approx(w, abs=1)   # width stays horizontal-ish
    assert np.linalg.norm(bl - tl) == pytest.approx(h, abs=1)
    assert tl[1] < bl[1] and tl[0] < tr[0]


def test_read_pklot_xml(tmp_path):
    xml = tmp_path / "a.xml"
    xml.write_text("""<?xml version="1.0"?>
<parking id="ufpr04">
  <space id="1" occupied="1">
    <rotatedRect><center x="50" y="50" /><size w="20" h="40" /><angle d="0" /></rotatedRect>
    <contour><point x="40" y="30" /><point x="60" y="30" /><point x="60" y="70" /><point x="40" y="70" /></contour>
  </space>
  <space id="2">
    <rotatedRect><center x="80" y="50" /><size w="20" h="40" /><angle d="10" /></rotatedRect>
  </space>
</parking>""")
    slots = read_pklot_xml(xml)
    assert [s["id"] for s in slots] == [1, 2]
    assert slots[0]["occupied"] == 1 and slots[1]["occupied"] is None
    assert slots[0]["polygon"].shape == (4, 2)
    assert slots[1]["polygon"].shape == (4, 2)  # rebuilt from the rotated rectangle


def test_read_cnr_camera_csv_rescales(tmp_path):
    f = tmp_path / "camera1.csv"
    f.write_text("SlotId,X,Y,W,H\n603,2592,1944,259.2,194.4\n")
    poly = read_cnr_camera_csv(f)[603]
    np.testing.assert_allclose(poly[0], [1000, 750], atol=1e-3)
    np.testing.assert_allclose(poly[2], [1100, 825], atol=1e-3)


def test_split_days_is_disjoint_stratified_and_deterministic():
    day_weather = {f"2012-10-{d:02d}": ["sunny"] * 5 for d in range(1, 13)}
    day_weather.update({f"2012-11-{d:02d}": ["rainy"] * 3 + ["cloudy"] for d in range(1, 5)})
    a, b = split_days(day_weather), split_days(day_weather)
    assert a == b and set(a) == set(day_weather)
    for weather in ["sunny", "rainy"]:
        assigned = {a[d] for d, w in day_weather.items() if w[0] == weather}
        assert assigned == {"train", "val", "test"}


def test_cnr_split_is_chronological():
    days = ["2016-01-08", "2015-11-12", "2015-11-16", "2015-11-20", "2015-12-03"]
    s = split_cnr_days(days, n_adapt=3)
    assert [d for d in sorted(s) if s[d] == "adapt"] == ["2015-11-12", "2015-11-16", "2015-11-20"]
    assert max(d for d in s if s[d] == "adapt") < min(d for d in s if s[d] == "later")


@pytest.mark.parametrize("preset", PRESETS)
def test_augment_keeps_shape_and_range(preset):
    x = torch.rand(8, 3, PATCH_SIZE, PATCH_SIZE)
    y = augment(x.clone(), preset)
    assert y.shape == x.shape and y.min() >= 0 and y.max() <= 1
    if preset == "none":
        assert torch.equal(x, y)


def test_class_weights_balance_classes():
    w = class_weights(np.array([0] * 30 + [1] * 10), "inverse")
    assert w.mean().item() == pytest.approx(1.0)
    assert (w[1] / w[0]).item() == pytest.approx(3.0)


def test_iterate_batches_covers_everything_once():
    seen = np.concatenate(list(iterate_batches(103, 10, True, np.random.default_rng(0))))
    assert sorted(seen.tolist()) == list(range(103))


def test_patch_writer_round_trip(tmp_path):
    w = PatchWriter("demo", tmp_path)
    imgs = np.random.default_rng(0).integers(0, 255, (5, PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
    w.add(imgs[:2], [{"label": i % 2, "weather": "sunny"} for i in range(2)])
    w.add(imgs[2:], [{"label": i % 2, "weather": "rainy"} for i in range(2, 5)])
    w.close()
    ps = load_patch_set("demo", tmp_path)
    np.testing.assert_array_equal(np.asarray(ps.images), imgs)
    assert ps.labels.tolist() == [0, 1, 0, 1, 0]
    assert len(ps.subset(ps.column("weather") == "rainy")) == 3
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("arch", ["mobilenet_v3_small", "malexnet"])
def test_onnx_export_matches_pytorch(tmp_path, arch):
    from parking.onnx_model import OnnxClassifier, export_onnx

    model = build_model(arch, pretrained=False).eval()
    path = export_onnx(model, {"arch": arch}, tmp_path / "m.onnx")
    x = torch.rand(5, 3, PATCH_SIZE, PATCH_SIZE)
    with torch.no_grad():
        ref = model(x).numpy()
    clf = OnnxClassifier(path)
    assert clf.config == {"arch": arch}
    np.testing.assert_allclose(clf(x.numpy()), ref, atol=1e-4)
