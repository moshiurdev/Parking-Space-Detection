"""FastAPI server for the parking occupancy demo.

Run:
    python -m app.server --checkpoint checkpoints/parking_mobilenetv3.onnx
    # or: uvicorn app.server:app --port 8000

The page plays a recorded sequence of lot frames (built by
``scripts/make_demo.py``): for each frame it asks the server to classify every
slot and draws the slot polygons in green (free) or red (occupied). A lot
image (or a video / camera frame) can also be uploaded: with a slot-polygon
JSON, or without one, in which case the slots are found automatically from
the painted line markings (``parking/autoslots.py``).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from parking.autoslots import detect_slots
from parking.predictor import SlotPredictor

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "parking_mobilenetv3.onnx"
DEFAULT_DEMO = ROOT / "data" / "demo" / "sequences.json"
DEFAULT_METRICS = ROOT / "results" / "mobilenetv3" / "metrics.json"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_SLOTS = 1000


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_predictor()  # load the model before the first request
    yield


app = FastAPI(title="Parking Space Occupancy", version="1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.middleware("http")
async def no_stale_page(request, call_next):
    """Make browsers revalidate the page and its scripts, so an updated demo is never mixed with a cached one."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@lru_cache(maxsize=1)
def get_predictor() -> SlotPredictor:
    checkpoint = Path(os.environ.get("PARKING_CHECKPOINT", DEFAULT_CHECKPOINT))
    if not checkpoint.exists():
        raise RuntimeError(f"checkpoint not found: {checkpoint} (see README: Pretrained weights)")
    return SlotPredictor(checkpoint, device=os.environ.get("PARKING_DEVICE", "cpu"))


@lru_cache(maxsize=1)
def get_sequences() -> dict:
    path = Path(os.environ.get("PARKING_DEMO", DEFAULT_DEMO))
    if not path.exists():
        return {}
    return {s["id"]: s for s in json.loads(path.read_text())["sequences"]}


def get_frame(seq_id: str, index: int) -> dict:
    seq = get_sequences().get(seq_id)
    if seq is None:
        raise HTTPException(404, f"unknown sequence {seq_id!r}")
    if not 0 <= index < len(seq["frames"]):
        raise HTTPException(404, "frame index out of range")
    return seq["frames"][index]


def resolve_image(rel: str) -> Path:
    """Frame paths are relative to the repo root; only files inside the repo or
    next to the sequences file may be served."""
    path = (ROOT / rel).resolve()
    allowed = {ROOT.resolve(), Path(os.environ.get("PARKING_DEMO", DEFAULT_DEMO)).resolve().parent}
    if not any(a in path.parents for a in allowed) or not path.is_file():
        raise HTTPException(404, "frame image not found")
    return path


def classify(rgb: np.ndarray, slots: list[dict]) -> dict:
    predictor = get_predictor()
    polygons = [np.asarray(s["polygon"], dtype=np.float32) for s in slots]
    prob, timing = predictor.predict(rgb, polygons)
    out = []
    for s, p in zip(slots, prob):
        item = {"id": s.get("id"), "polygon": s["polygon"], "p_occupied": round(float(p), 4),
                "occupied": bool(p >= predictor.threshold)}
        if s.get("label") is not None:
            item["label"] = int(s["label"])
        out.append(item)
    n_occ = sum(o["occupied"] for o in out)
    result = {"slots": out, "total": len(out), "occupied": n_occ, "free": len(out) - n_occ,
              "width": rgb.shape[1], "height": rgb.shape[0],
              "timing_ms": {k: round(v, 2) for k, v in timing.items()}}
    labelled = [o for o in out if "label" in o]
    if labelled:
        correct = sum(o["occupied"] == bool(o["label"]) for o in labelled)
        result["ground_truth"] = {"free": sum(1 - o["label"] for o in labelled), "labelled": len(labelled),
                                  "correct": correct, "accuracy": correct / len(labelled)}
    return result


def parse_slots(raw: str) -> list[dict]:
    """Accept ``[[x, y], ...]`` polygons or ``{"slots": [{"id", "polygon"}]}``."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"slots JSON is not valid JSON: {e}")
    if isinstance(data, dict):
        data = data.get("slots", [])
    if not isinstance(data, list) or not data:
        raise HTTPException(400, "slots JSON must be a non-empty list of polygons")
    if len(data) > MAX_SLOTS:
        raise HTTPException(400, f"too many slots (max {MAX_SLOTS})")
    slots = []
    for i, s in enumerate(data):
        poly = s.get("polygon") if isinstance(s, dict) else s
        try:
            arr = np.asarray(poly, dtype=np.float32)
        except (TypeError, ValueError):
            arr = np.zeros(0)
        if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 3 or not np.isfinite(arr).all():
            raise HTTPException(400, f"slot {i}: polygon must be a list of at least 3 [x, y] points")
        slots.append({"id": s.get("id", i) if isinstance(s, dict) else i, "polygon": arr.tolist()})
    return slots


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info() -> dict:
    cfg = get_predictor().config
    out = {"model": {"name": cfg["name"], "arch": cfg["arch"], "input_size": cfg["input_size"],
                     "params": cfg.get("params")}}
    metrics_path = Path(os.environ.get("PARKING_METRICS", DEFAULT_METRICS))
    if metrics_path.exists():
        sets = json.loads(metrics_path.read_text())["sets"]
        out["test_accuracy"] = {k: sets[k]["overall"]["accuracy"] for k in ["pklot_test", "pklot_unseen", "cnr_ext"]}
    return out


@app.get("/api/sequences")
def sequences() -> list[dict]:
    return [{"id": s["id"], "title": s["title"], "description": s.get("description", ""),
             "frames": len(s["frames"]), "slots": len(s["frames"][0]["slots"]) if s["frames"] else 0}
            for s in get_sequences().values()]


@app.get("/api/sequences/{seq_id}/{index}/image")
def frame_image(seq_id: str, index: int) -> FileResponse:
    return FileResponse(resolve_image(get_frame(seq_id, index)["image"]), media_type="image/jpeg")


@app.get("/api/sequences/{seq_id}/{index}")
def frame_predict(seq_id: str, index: int) -> dict:
    frame = get_frame(seq_id, index)
    bgr = cv2.imread(str(resolve_image(frame["image"])))
    if bgr is None:
        raise HTTPException(500, "could not read frame image")
    result = classify(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), frame["slots"])
    result.update({"index": index, "time": frame.get("time"), "weather": frame.get("weather")})
    return result


@app.post("/api/analyze")
async def analyze(image: UploadFile = File(...), slots: UploadFile | None = File(None),
                  slots_json: str | None = Form(None)) -> dict:
    data = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "image too large (max 10 MB)")
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "could not decode image")
    if slots is not None:
        raw = (await slots.read(MAX_UPLOAD_BYTES)).decode("utf-8", errors="replace")
    elif slots_json:
        raw = slots_json
    else:
        return analyze_auto(bgr)
    return classify(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), parse_slots(raw))


def analyze_auto(bgr: np.ndarray) -> dict:
    """Find the slots from the line markings, then classify them."""
    t0 = time.perf_counter()
    polygons = detect_slots(bgr)[:MAX_SLOTS]
    detect_ms = (time.perf_counter() - t0) * 1000
    if not polygons:
        raise HTTPException(422, "No parking slot lines were found automatically. This works for lots seen "
                                 "from above with painted stall lines; otherwise draw the slots by hand.")
    slots = [{"id": i + 1, "polygon": np.round(p, 1).tolist()} for i, p in enumerate(polygons)]
    result = classify(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), slots)
    result["auto"] = True
    result["timing_ms"]["detect_ms"] = round(detect_ms, 2)
    return result


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


if __name__ == "__main__":
    import uvicorn

    p = argparse.ArgumentParser(description="Serve the parking occupancy web demo")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    os.environ["PARKING_CHECKPOINT"] = str(args.checkpoint)
    uvicorn.run(app, host=args.host, port=args.port)
