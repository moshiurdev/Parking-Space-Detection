"""Web API, using a random tiny checkpoint and a synthetic demo sequence."""

import io
import json

import cv2
import pytest
from fastapi.testclient import TestClient

from conftest import lot_image


@pytest.fixture(scope="module")
def client(tiny_checkpoint, demo_dir, monkeypatch_module):
    monkeypatch_module.setenv("PARKING_CHECKPOINT", str(tiny_checkpoint))
    monkeypatch_module.setenv("PARKING_DEMO", str(demo_dir / "sequences.json"))
    monkeypatch_module.setenv("PARKING_METRICS", str(demo_dir / "missing.json"))
    from app import server

    server.get_predictor.cache_clear()
    server.get_sequences.cache_clear()
    with TestClient(server.app) as c:
        yield c
    server.get_predictor.cache_clear()
    server.get_sequences.cache_clear()


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def jpeg_bytes() -> bytes:
    img, _ = lot_image()
    return cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tobytes()


def test_index_and_info(client):
    assert client.get("/").status_code == 200
    info = client.get("/api/info").json()
    assert info["model"]["arch"] == "malexnet"


def test_sequences_and_frames(client):
    seqs = client.get("/api/sequences").json()
    assert seqs == [{"id": "test", "title": "Test lot", "description": "synthetic", "frames": 3, "slots": 2}]
    r = client.get("/api/sequences/test/1")
    assert r.status_code == 200
    res = r.json()
    assert res["total"] == 2 and res["free"] + res["occupied"] == 2
    assert res["ground_truth"]["free"] == 1 and res["ground_truth"]["labelled"] == 2
    assert set(res["timing_ms"]) == {"crop_ms", "model_ms", "total_ms"}
    assert all(0 <= s["p_occupied"] <= 1 for s in res["slots"])
    img = client.get("/api/sequences/test/1/image")
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"


def test_frame_errors(client):
    assert client.get("/api/sequences/nope/0").status_code == 404
    assert client.get("/api/sequences/test/99").status_code == 404


def test_analyze_upload(client):
    _, slots = lot_image()
    body = json.dumps({"slots": [{"id": 7, "polygon": s.tolist()} for s in slots]})
    r = client.post("/api/analyze", files={"image": ("lot.jpg", jpeg_bytes(), "image/jpeg"),
                                           "slots": ("slots.json", io.BytesIO(body.encode()), "application/json")})
    assert r.status_code == 200
    res = r.json()
    assert res["total"] == 2 and res["slots"][0]["id"] == 7 and "ground_truth" not in res
    # plain list of polygons in a form field
    r = client.post("/api/analyze", files={"image": ("lot.jpg", jpeg_bytes(), "image/jpeg")},
                    data={"slots_json": json.dumps([s.tolist() for s in slots])})
    assert r.status_code == 200 and r.json()["total"] == 2


@pytest.mark.parametrize("slots_json", ["not json", "[]", "[[[0, 0], [1, 1]]]", '[{"polygon": "x"}]'])
def test_analyze_rejects_bad_slots(client, slots_json):
    r = client.post("/api/analyze", files={"image": ("lot.jpg", jpeg_bytes(), "image/jpeg")},
                    data={"slots_json": slots_json})
    assert r.status_code == 400


def test_analyze_rejects_bad_image(client):
    r = client.post("/api/analyze", files={"image": ("lot.jpg", b"nope", "image/jpeg")},
                    data={"slots_json": "[[[0,0],[1,0],[1,1]]]"})
    assert r.status_code == 400
