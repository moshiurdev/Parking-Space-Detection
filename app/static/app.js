"use strict";

const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const ctx = canvas.getContext("2d");
const video = $("video");

let mode = "replay";

// ---------------------------------------------------------------- drawing

const fmtMs = (v) => (v == null ? "–" : `${v.toFixed(1)} ms`);
const fmtPct = (v) => (v == null ? "–" : `${(v * 100).toFixed(1)}%`);

function tracePolygon(pts) {
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (const [x, y] of pts.slice(1)) ctx.lineTo(x, y);
  ctx.closePath();
}

function lineWidth() {
  return Math.max(1.5, canvas.width / 600);
}

/** Draw a frame (image, video or canvas source) with classified slots on top. */
function drawResult(source, w, h, result, showErr) {
  canvas.width = w;
  canvas.height = h;
  ctx.drawImage(source, 0, 0, w, h);
  const lw = lineWidth();
  for (const s of result.slots) {
    tracePolygon(s.polygon);
    const rgb = s.occupied ? "231,76,60" : "46,204,113";
    ctx.fillStyle = `rgba(${rgb},0.28)`;
    ctx.fill();
    ctx.lineWidth = lw;
    ctx.strokeStyle = `rgb(${rgb})`;
    ctx.stroke();
    if (showErr && s.label != null && s.occupied !== Boolean(s.label)) {
      ctx.lineWidth = lw * 2.5;
      ctx.strokeStyle = "#f1c40f";
      ctx.stroke();
    }
  }
}

function showStats(result, rtt, run) {
  $("free").textContent = result ? result.free : "–";
  $("occupied").textContent = result ? result.occupied : "–";
  $("total").textContent = result ? result.total : "–";
  $("t-crop").textContent = fmtMs(result?.timing_ms.crop_ms);
  $("t-model").textContent = fmtMs(result?.timing_ms.model_ms);
  $("t-rtt").textContent = fmtMs(rtt);
  const gt = result?.ground_truth;
  $("gt-free").textContent = gt ? gt.free : "–";
  $("gt-acc").textContent = gt ? fmtPct(gt.accuracy) : "–";
  $("run-acc").textContent = run && run.total ? `${fmtPct(run.correct / run.total)} (${run.total} slots)` : "–";
}

function setStatus(text, isError = false) {
  const el = $("status");
  el.textContent = text;
  el.className = isError ? "status error" : "status";
}

function showEmpty(text) {
  $("empty").textContent = text;
  $("empty").classList.toggle("hidden", !text);
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("could not load image"));
    img.src = src;
  });
}

// ---------------------------------------------------------------- dataset replay

const replay = {
  sequences: [],
  seq: null,
  index: 0,
  playing: false,
  busy: false,
  timer: null,
  last: null,
  run: { correct: 0, total: 0 },
};

async function showFrame(index) {
  if (!replay.seq || replay.busy) return;
  replay.busy = true;
  const seq = replay.seq;
  const t0 = performance.now();
  try {
    const base = `/api/sequences/${encodeURIComponent(seq.id)}/${index}`;
    const [res, img] = await Promise.all([
      fetch(base).then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`)))),
      loadImage(`${base}/image`),
    ]);
    if (seq !== replay.seq || mode !== "replay") return;
    const rtt = performance.now() - t0;
    replay.index = index;
    if (res.ground_truth) {
      replay.run.correct += res.ground_truth.correct;
      replay.run.total += res.ground_truth.labelled;
    }
    replay.last = { img, res };
    drawResult(img, img.naturalWidth, img.naturalHeight, res, $("show-gt").checked);
    showStats(res, rtt, replay.run);
    $("scrub").value = index;
    $("frame-label").textContent = `Frame ${index + 1} / ${seq.frames}`;
    $("frame-meta").textContent = [res.time, res.weather].filter(Boolean).join(" · ");
  } catch (err) {
    console.error(err);
    stopReplay();
    setStatus(`Error: ${err.message}`, true);
  } finally {
    replay.busy = false;
  }
}

function scheduleNext() {
  if (!replay.playing) return;
  const fps = Number($("fps").value);
  const started = performance.now();
  const next = (replay.index + 1) % replay.seq.frames;
  if (next === 0) replay.run = { correct: 0, total: 0 };
  showFrame(next).then(() => {
    const wait = Math.max(0, 1000 / fps - (performance.now() - started));
    replay.timer = setTimeout(scheduleNext, wait);
  });
}

function playReplay() {
  if (!replay.seq) return;
  replay.playing = true;
  $("play").textContent = "Pause";
  scheduleNext();
}

function stopReplay() {
  replay.playing = false;
  clearTimeout(replay.timer);
  $("play").textContent = "Play";
}

function selectSequence(id, frame = 0) {
  stopReplay();
  replay.seq = replay.sequences.find((s) => s.id === id) || null;
  replay.run = { correct: 0, total: 0 };
  if (!replay.seq) return;
  $("seq-desc").textContent = replay.seq.description;
  $("scrub").max = replay.seq.frames - 1;
  showFrame(Math.min(Math.max(frame, 0), replay.seq.frames - 1));
}

// ---------------------------------------------------------------- your own image / video / camera

const custom = {
  source: null,      // { kind: "image" | "video" | "camera", el, w, h }
  slots: [],         // polygons in source pixel coordinates
  draft: [],         // points of the slot being drawn
  drawing: false,
  result: null,      // last /api/analyze result for the current slots
  live: false,
  busy: false,
  raf: null,
  objectUrl: null,
};
const grab = document.createElement("canvas");

function stopSource() {
  stopLive();
  cancelAnimationFrame(custom.raf);
  if (custom.source?.kind === "camera") custom.source.el.srcObject?.getTracks().forEach((t) => t.stop());
  if (custom.source?.kind !== "image") {
    video.pause();
    video.removeAttribute("src");
    video.srcObject = null;
  }
  if (custom.objectUrl) URL.revokeObjectURL(custom.objectUrl);
  custom.objectUrl = null;
  custom.source = null;
}

function setSource(kind, el, w, h) {
  custom.source = { kind, el, w, h };
  custom.result = null;
  $("live").classList.toggle("hidden", kind === "image");
  $("analyse").textContent = kind === "image" ? "Analyse" : "Analyse frame";
  showEmpty("");
  if (kind === "image") renderCustom();
  else renderLoop();
  if (!custom.slots.length) startDrawing();
  updateHelp();
}

function renderLoop() {
  renderCustom();
  custom.raf = requestAnimationFrame(renderLoop);
}

function renderCustom() {
  if (mode !== "custom") return;
  const src = custom.source;
  if (!src) return;
  const r = custom.result;
  if (r && r.slots.length === custom.slots.length) {
    drawResult(src.el, src.w, src.h, r, false);
  } else {
    canvas.width = src.w;
    canvas.height = src.h;
    ctx.drawImage(src.el, 0, 0, src.w, src.h);
    ctx.setLineDash([6, 4]);
    ctx.lineWidth = lineWidth();
    ctx.strokeStyle = "#38bdf8";
    for (const poly of custom.slots) {
      tracePolygon(poly);
      ctx.stroke();
    }
    ctx.setLineDash([]);
  }
  // slot being drawn
  if (custom.draft.length) {
    ctx.strokeStyle = "#38bdf8";
    ctx.fillStyle = "#38bdf8";
    ctx.lineWidth = lineWidth();
    ctx.beginPath();
    custom.draft.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.stroke();
    for (const [x, y] of custom.draft) {
      ctx.beginPath();
      ctx.arc(x, y, lineWidth() * 2, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}

function slotsChanged() {
  custom.result = null;
  showStats(null);
  renderCustom();
  updateHelp();
}

function updateHelp() {
  const n = custom.slots.length;
  let text;
  if (!custom.source) text = "Choose an image, a video or your camera.";
  else if (custom.drawing) text = `Drawing: click the 4 corners of each slot (${n} slot${n === 1 ? "" : "s"} so far). Esc cancels the current slot.`;
  else if (!n) text = "Draw one polygon per parking slot (Draw slots, then click its 4 corners) or load a slots JSON.";
  else text = `${n} slot${n === 1 ? "" : "s"}. Press ${custom.source.kind === "image" ? "Analyse" : "Analyse frame or Start live"}.`;
  $("custom-help").textContent = text;
}

function startDrawing() {
  custom.drawing = true;
  $("draw").classList.add("active-tool");
  canvas.classList.add("drawing");
  updateHelp();
}

function stopDrawing() {
  custom.drawing = false;
  custom.draft = [];
  $("draw").classList.remove("active-tool");
  canvas.classList.remove("drawing");
  renderCustom();
  updateHelp();
}

function canvasPoint(e) {
  const rect = canvas.getBoundingClientRect();
  return [
    Math.round(((e.clientX - rect.left) * canvas.width) / rect.width),
    Math.round(((e.clientY - rect.top) * canvas.height) / rect.height),
  ];
}

canvas.addEventListener("click", (e) => {
  if (mode !== "custom" || !custom.drawing || !custom.source) return;
  custom.draft.push(canvasPoint(e));
  if (custom.draft.length === 4) {
    custom.slots.push(custom.draft);
    custom.draft = [];
    slotsChanged();
  } else {
    renderCustom();
  }
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && custom.draft.length) {
    custom.draft = [];
    renderCustom();
  }
});

async function analyseCurrentFrame() {
  const src = custom.source;
  if (!src || !custom.slots.length || custom.busy) return false;
  custom.busy = true;
  const t0 = performance.now();
  try {
    let blob;
    if (src.kind === "image" && $("src-image").files[0]) {
      blob = $("src-image").files[0];
    } else {
      grab.width = src.w;
      grab.height = src.h;
      grab.getContext("2d").drawImage(src.el, 0, 0, src.w, src.h);
      blob = await new Promise((resolve) => grab.toBlob(resolve, "image/jpeg", 0.9));
    }
    const form = new FormData();
    form.append("image", blob, "frame.jpg");
    form.append("slots_json", JSON.stringify(custom.slots));
    const r = await fetch("/api/analyze", { method: "POST", body: form });
    const res = await r.json();
    if (!r.ok) throw new Error(res.detail || `HTTP ${r.status}`);
    custom.result = res;
    renderCustom();
    showStats(res, performance.now() - t0);
    setStatus(`${res.free} of ${res.total} slots free.`);
    return true;
  } catch (err) {
    setStatus(err.message, true);
    stopLive();
    return false;
  } finally {
    custom.busy = false;
  }
}

async function liveLoop() {
  while (custom.live && mode === "custom") {
    const started = performance.now();
    if (!(await analyseCurrentFrame())) break;
    // at most ~8 analyses per second
    await new Promise((r) => setTimeout(r, Math.max(0, 125 - (performance.now() - started))));
  }
}

function startLive() {
  if (!custom.slots.length) {
    setStatus("Draw or load slots first.", true);
    return;
  }
  stopDrawing();
  custom.live = true;
  $("live").textContent = "Stop live";
  liveLoop();
}

function stopLive() {
  custom.live = false;
  $("live").textContent = "Start live";
}

function parseSlotsJson(text) {
  let data = JSON.parse(text);
  if (!Array.isArray(data)) data = data.slots || [];
  return data.map((s) => (Array.isArray(s) ? s : s.polygon)).filter((p) => Array.isArray(p) && p.length >= 3);
}

$("src-image").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  stopSource();
  custom.objectUrl = URL.createObjectURL(file);
  try {
    const img = await loadImage(custom.objectUrl);
    setSource("image", img, img.naturalWidth, img.naturalHeight);
    setStatus("");
  } catch (err) {
    setStatus(err.message, true);
  }
});

$("src-video").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  stopSource();
  custom.objectUrl = URL.createObjectURL(file);
  video.srcObject = null;
  video.src = custom.objectUrl;
  video.onloadeddata = () => {
    video.onloadeddata = null;
    setSource("video", video, video.videoWidth, video.videoHeight);
    video.play();
    setStatus("");
  };
  video.onerror = () => setStatus("This browser cannot play that video file.", true);
});

$("src-camera").addEventListener("click", async () => {
  stopSource();
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("Camera access needs localhost or HTTPS.", true);
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 } }, audio: false });
    video.removeAttribute("src");
    video.srcObject = stream;
    await video.play();
    setSource("camera", video, video.videoWidth, video.videoHeight);
    setStatus("Camera on. Point it at a parking area (or a printed lot photo) and draw the slots.");
  } catch (err) {
    setStatus(`Camera: ${err.message}`, true);
  }
});

$("draw").addEventListener("click", () => (custom.drawing ? stopDrawing() : startDrawing()));
$("undo").addEventListener("click", () => {
  if (custom.draft.length) custom.draft.pop();
  else custom.slots.pop();
  slotsChanged();
});
$("clear").addEventListener("click", () => {
  custom.slots = [];
  custom.draft = [];
  slotsChanged();
});
$("slots-file").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  try {
    custom.slots = parseSlotsJson(await file.text());
    stopDrawing();
    slotsChanged();
    setStatus(`Loaded ${custom.slots.length} slots.`);
  } catch (err) {
    setStatus(`Slots JSON: ${err.message}`, true);
  }
  e.target.value = "";
});
$("save").addEventListener("click", () => {
  const body = JSON.stringify({ slots: custom.slots.map((p, i) => ({ id: i + 1, polygon: p })) }, null, 1);
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([body], { type: "application/json" }));
  a.download = "slots.json";
  a.click();
  URL.revokeObjectURL(a.href);
});
$("analyse").addEventListener("click", () => {
  if (!custom.slots.length) {
    setStatus("Draw or load slots first.", true);
    return;
  }
  stopDrawing();
  analyseCurrentFrame();
});
$("live").addEventListener("click", () => (custom.live ? stopLive() : startLive()));

// ---------------------------------------------------------------- tabs and start-up

function setMode(next) {
  mode = next;
  const isReplay = mode === "replay";
  $("tab-replay").classList.toggle("active", isReplay);
  $("tab-custom").classList.toggle("active", !isReplay);
  $("tab-replay").setAttribute("aria-selected", isReplay);
  $("tab-custom").setAttribute("aria-selected", !isReplay);
  $("replay-controls").classList.toggle("hidden", !isReplay);
  $("replay-footer").classList.toggle("hidden", !isReplay);
  $("custom-controls").classList.toggle("hidden", isReplay);
  setStatus("");
  if (isReplay) {
    stopSource();
    stopDrawing();
    showEmpty(replay.sequences.length ? "" : "No demo sequences found. Run python scripts/make_demo.py, or use your own image.");
    if (replay.last) {
      drawResult(replay.last.img, replay.last.img.naturalWidth, replay.last.img.naturalHeight,
                 replay.last.res, $("show-gt").checked);
      showStats(replay.last.res, null, replay.run);
    }
  } else {
    stopReplay();
    showStats(null);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    showEmpty(custom.source ? "" : "Choose an image, a video file or your camera above.");
    if (custom.source) setSource(custom.source.kind, custom.source.el, custom.source.w, custom.source.h);
    updateHelp();
  }
}

async function init() {
  fetch("/api/info").then((r) => r.json()).then((info) => {
    const m = info.model;
    let text = `Model: ${m.name} (${m.arch}, ${(m.params / 1e6).toFixed(2)} M params, ${m.input_size}px input).`;
    if (info.test_accuracy) {
      const a = info.test_accuracy;
      const p2 = (v) => `${(v * 100).toFixed(2)}%`;
      text += ` Test accuracy: PKLot held-out days ${p2(a.pklot_test)}, unseen lot ${p2(a.pklot_unseen)}, CNRPark-EXT ${p2(a.cnr_ext)}.`;
    }
    $("model-info").textContent = text;
  });

  replay.sequences = await fetch("/api/sequences").then((r) => r.json());
  const sel = $("sequence");
  for (const s of replay.sequences) {
    const opt = document.createElement("option");
    opt.value = s.id;
    opt.textContent = `${s.title} (${s.frames} frames, ${s.slots} slots)`;
    sel.appendChild(opt);
  }
  if (!replay.sequences.length) {
    $("play").disabled = $("step").disabled = true;
    setMode("custom");
    return;
  }
  // optional deep link: #seq=<id>&frame=<n>&errors=1
  const hash = new URLSearchParams(location.hash.slice(1));
  const start = replay.sequences.find((s) => s.id === hash.get("seq")) || replay.sequences[0];
  $("show-gt").checked = hash.get("errors") === "1";
  sel.value = start.id;
  selectSequence(start.id, Number(hash.get("frame")) || 0);
}

$("tab-replay").addEventListener("click", () => setMode("replay"));
$("tab-custom").addEventListener("click", () => setMode("custom"));
$("sequence").addEventListener("change", (e) => selectSequence(e.target.value));
$("play").addEventListener("click", () => (replay.playing ? stopReplay() : playReplay()));
$("step").addEventListener("click", () => {
  stopReplay();
  if (replay.seq) showFrame((replay.index + 1) % replay.seq.frames);
});
$("scrub").addEventListener("input", (e) => {
  stopReplay();
  showFrame(Number(e.target.value));
});
$("show-gt").addEventListener("change", () => {
  const l = replay.last;
  if (l) drawResult(l.img, l.img.naturalWidth, l.img.naturalHeight, l.res, $("show-gt").checked);
});

init();
