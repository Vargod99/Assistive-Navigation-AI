#!/usr/bin/env python3
"""End-to-end low-latency navigation pipeline.

frame -> [resize] -> YOLOv3 -> NMS -> MobileNetV2 posture (person crops)
      -> clock direction + pinhole distance -> path planner -> spoken instruction

Examples
--------
    python pipeline.py --image test_scene.jpg                 # single image + latency report
    python pipeline.py --image test_scene.jpg --runs 20 --save out.jpg
    python pipeline.py --camera 0 --speak                     # live webcam (macOS `say`)
    python pipeline.py --video walk.mp4 --model yolov3-tiny
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import statistics
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from geometry import CameraModel, SpatialInfo, format_distance, localize
from models import PostureClassifier, YOLOv3Detector, select_device, sync

log = logging.getLogger("pipeline")
ROOT = Path(__file__).resolve().parent

# Classes that physically obstruct a walking path / matter to a pedestrian.
OBSTACLE_CLASSES = {
    "person", "chair", "couch", "bed", "dining table", "toilet", "tv", "refrigerator",
    "oven", "sink", "potted plant", "bench", "suitcase", "backpack", "handbag", "dog", "cat",
    "bicycle", "motorcycle", "car", "bus", "truck", "fire hydrant", "stop sign",
    "parking meter", "umbrella", "laptop", "microwave",
}

DISPLAY_NAMES = {"tv": "TV"}

STAGES = ("preprocess", "detect", "nms", "posture", "localize_plan")


@dataclass
class ObjectReport:
    label: str
    confidence: float
    box: tuple[int, int, int, int]
    spatial: SpatialInfo
    posture: str | None = None
    posture_conf: float | None = None
    approaching: bool = False
    in_path: bool = False
    track_id: int | None = None

    def describe(self, say_posture: bool) -> str:
        name = DISPLAY_NAMES.get(self.label, self.label)
        if self.label == "person":
            bits = []
            if say_posture and self.posture:
                bits.append(self.posture)
            if self.approaching:
                bits.append("approaching")
            if bits:
                name = f"person {', '.join(bits)}"
        dist = format_distance(self.spatial.distance_m, self.spatial.distance_is_upper_bound)
        return f"{name} at {self.spatial.clock_str}, {dist}"


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


@dataclass
class FrameResult:
    instruction: str
    announcements: list[str]
    objects: list[ObjectReport]
    action: str          # "clear" | "left" | "right" | "blocked"
    urgency: int         # 0 clear, 1 ahead, 2 caution, 3 stop
    timings_ms: dict[str, float]
    budget_ms: float
    image_size: tuple[int, int]

    @property
    def total_ms(self) -> float:
        return self.timings_ms["total"]

    @property
    def within_budget(self) -> bool:
        return self.total_ms < self.budget_ms

    def to_dict(self) -> dict:
        d = asdict(self)
        d["within_budget"] = self.within_budget
        return d


def _iou(a, b) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


class ApproachTracker:
    """Greedy IoU tracker; marks a person 'approaching' when range shrinks faster
    than ``speed_mps`` over at least ``min_window_s`` seconds."""

    def __init__(self, iou_thr: float = 0.3, speed_mps: float = 0.35,
                 min_window_s: float = 0.5, max_age_s: float = 1.0):
        self.iou_thr, self.speed, self.window, self.max_age = iou_thr, speed_mps, min_window_s, max_age_s
        self.tracks: dict[int, dict] = {}
        self._next = 0

    def update(self, objs: list[ObjectReport], t: float) -> None:
        self.tracks = {k: v for k, v in self.tracks.items() if t - v["t"] <= self.max_age}
        free = set(self.tracks)
        for o in sorted((o for o in objs if o.label == "person"), key=lambda o: -o.confidence):
            best, best_iou = None, self.iou_thr
            for tid in free:
                iou = _iou(o.box, self.tracks[tid]["box"])
                if iou > best_iou:
                    best, best_iou = tid, iou
            if best is None:
                best = self._next
                self._next += 1
                self.tracks[best] = {"hist": deque(maxlen=30)}
            else:
                free.discard(best)
            tr = self.tracks[best]
            tr.update(box=o.box, t=t)
            o.track_id = best
            if o.spatial.distance_m is not None:
                tr["hist"].append((t, o.spatial.distance_m))
            h = tr["hist"]
            if len(h) >= 3 and h[-1][0] - h[0][0] >= self.window:
                ts, ds = np.array([p[0] for p in h]), np.array([p[1] for p in h])
                slope = np.polyfit(ts - ts[0], ds, 1)[0]  # m/s, negative = closing
                o.approaching = bool(slope < -self.speed)


class NavigationPlanner:
    """Turns localised objects into one movement instruction.

    A straight-ahead walking corridor of half-width ``corridor_m`` is checked
    out to ``lookahead_m``. If blocked, the planner tests a corridor just to
    the left and right of the nearest blocker and picks the clear one needing
    the smaller sideways shift. A side corridor is only eligible if part of it
    is actually inside the camera's view at the blocker's distance - the
    planner never steers the user into space it cannot see.
    """

    def __init__(self, corridor_m: float = 0.45, lookahead_m: float = 3.0,
                 stop_m: float = 1.0, caution_m: float = 2.0, max_announcements: int = 3,
                 margin_m: float = 0.15, hfov_deg: float = 60.0):
        self.hw, self.look, self.stop, self.caution = corridor_m, lookahead_m, stop_m, caution_m
        self.tan_half = math.tan(math.radians(hfov_deg) / 2)
        self.max_ann, self.margin = max_announcements, margin_m

    @staticmethod
    def _overlaps(lat: tuple[float, float], lo: float, hi: float) -> bool:
        return lat[1] > lo and lat[0] < hi

    def _corridor_clear(self, objs: list[ObjectReport], centre: float, skip=None) -> bool:
        lo, hi = centre - self.hw, centre + self.hw
        return not any(
            o is not skip and o.spatial.distance_m is not None and o.spatial.distance_m <= self.look
            and self._overlaps(o.spatial.lateral_m, lo, hi)
            for o in objs
        )

    def plan(self, objs: list[ObjectReport], say_posture: bool) -> tuple[str, list[str], str, int]:
        ranged = [o for o in objs if o.spatial.distance_m is not None and o.spatial.lateral_m is not None]
        for o in ranged:
            o.in_path = o.spatial.distance_m <= self.look and self._overlaps(o.spatial.lateral_m, -self.hw, self.hw)
        blockers = sorted((o for o in ranged if o.in_path), key=lambda o: o.spatial.distance_m)

        def by_range(o: ObjectReport):
            return o.spatial.distance_m if o.spatial.distance_m is not None else float("inf")

        if not blockers:
            instruction, action, urgency = "Path ahead is clear.", "clear", 0
            announce = sorted(objs, key=by_range)[: self.max_ann]
        else:
            b = blockers[0]
            d = b.spatial.distance_m
            prefix = "Stop." if d < self.stop else ("Caution." if d < self.caution else "Ahead:")
            urgency = 3 if d < self.stop else (2 if d < self.caution else 1)
            left_c = b.spatial.lateral_m[0] - self.hw - self.margin
            right_c = b.spatial.lateral_m[1] + self.hw + self.margin
            view_half = d * self.tan_half  # visible half-width at the blocker's range
            options = [(abs(c), side) for c, side in ((left_c, "left"), (right_c, "right"))
                       if abs(c) - self.hw < view_half and self._corridor_clear(ranged, c, skip=b)]
            if options:
                shift, side = min(options)
                verb = "Step" if shift > 0.9 else "Veer"
                action, advice = side, f"{verb} {side}."
            else:
                action, advice = "blocked", "Path blocked. Stop and turn to find another route."
                if prefix != "Stop.":
                    prefix, urgency = "Caution.", 2
            instruction = f"{prefix} {_cap(b.describe(say_posture))}. {advice}"
            announce = [o for o in sorted(objs, key=by_range) if o is not b][: self.max_ann - 1]
        return instruction, [_cap(o.describe(say_posture)) for o in announce], action, urgency


class InstructionSmoother:
    """Hysteresis for video streams so guidance doesn't flicker frame-to-frame.

    A new (action, urgency) state must persist for ``persist`` consecutive
    frames before it is announced - except an escalation to "Stop", which is
    announced immediately. The stable state is re-announced every
    ``repeat_s`` seconds so the user knows the system is alive.
    """

    def __init__(self, persist: int = 3, repeat_s: float = 6.0):
        self.persist, self.repeat_s = persist, repeat_s
        self.stable: tuple[str, int] | None = None
        self.candidate: tuple[str, int] | None = None
        self.count = 0
        self.last_emit = -1e9

    def update(self, res: FrameResult, t: float) -> str | None:
        state = (res.action, res.urgency)
        if state == self.stable:
            self.candidate, self.count = None, 0
            if t - self.last_emit >= self.repeat_s:
                self.last_emit = t
                return res.instruction
            return None
        if state == self.candidate:
            self.count += 1
        else:
            self.candidate, self.count = state, 1
        stable_urg = self.stable[1] if self.stable else -1
        if self.count >= self.persist or (res.urgency == 3 and res.urgency > stable_urg) or self.stable is None:
            self.stable, self.candidate, self.count, self.last_emit = state, None, 0, t
            return res.instruction
        return None


def resolve_yolo(weights_dir: Path, model: str) -> tuple[str, Path, Path]:
    candidates = ["yolov3", "yolov3-tiny"] if model == "auto" else [model]
    for name in candidates:
        cfg, w = weights_dir / f"{name}.cfg", weights_dir / f"{name}.weights"
        if cfg.is_file() and w.is_file():
            if model == "auto" and name != candidates[0]:
                log.warning("yolov3.weights not found - falling back to %s", name)
            return name, cfg, w
    raise FileNotFoundError(f"No YOLO weights for {model!r} in {weights_dir}. Run `python setup_weights.py`.")


class NavigationPipeline:
    def __init__(
        self,
        weights_dir: str | Path = ROOT / "weights",
        model: str = "auto",
        input_size: int = 416,
        device: str = "auto",
        hfov_deg: float = 60.0,
        clock_mode: str = "angular",
        conf_threshold: float = 0.4,
        nms_threshold: float = 0.45,
        posture_checkpoint: str | Path | None = None,
        posture_size: int | None = None,
        max_posture_crops: int = 4,
        max_frame_side: int = 1280,
        budget_ms: float = 300.0,
        announce_all: bool = False,
        half: bool = False,
        warmup: bool = True,
    ):
        weights_dir = Path(weights_dir)
        self.device = select_device(device)
        self.hfov, self.clock_mode = hfov_deg, clock_mode
        self.max_side, self.budget_ms = max_frame_side, budget_ms
        self.max_crops, self.announce_all = max_posture_crops, announce_all

        t0 = time.perf_counter()
        self.model_name, cfg, w = resolve_yolo(weights_dir, model)
        self.detector = YOLOv3Detector(cfg, w, weights_dir / "coco.names", input_size=input_size,
                                       conf_threshold=conf_threshold, nms_threshold=nms_threshold,
                                       device=self.device, half=half)
        ckpt = Path(posture_checkpoint) if posture_checkpoint else weights_dir / "mobilenetv2_posture.pth"
        self.posture = PostureClassifier(ckpt, self.device, posture_size) if ckpt.is_file() else None
        if self.posture is None:
            log.warning("posture checkpoint %s missing - posture disabled", ckpt)
        elif not self.posture.trained:
            log.warning("posture checkpoint has an UNTRAINED head - classifier runs (for timing) "
                        "but its labels are not announced. Fine-tune with train_posture.py.")
        self.load_s = time.perf_counter() - t0

        self.planner = NavigationPlanner(hfov_deg=hfov_deg)
        self.tracker = ApproachTracker()
        self._cams: dict[tuple[int, int], CameraModel] = {}
        self.cold_ms: float | None = None
        if warmup:
            self._warmup()

    @property
    def say_posture(self) -> bool:
        return self.posture is not None and self.posture.trained

    def _warmup(self, n: int = 3) -> None:
        """Prime kernels / MPS graphs so the first real frame is not penalised."""
        rng = np.random.default_rng(0)
        dummy = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        for i in range(n):
            t = time.perf_counter()
            self.detector.detect(dummy)
            if self.posture is not None:
                self.posture.predict(dummy, [(100, 50, 120, 300)])
            sync(self.device)
            if i == 0:
                self.cold_ms = (time.perf_counter() - t) * 1000
        # MPS compiles one graph per input shape: prime every posture batch size
        # we can emit so no real frame pays the compile cost.
        if self.posture is not None:
            for bs in range(2, self.max_crops + 1):
                for _ in range(2):
                    self.posture.predict(dummy, [(100, 50, 120, 300)] * bs)
            sync(self.device)

    def _camera(self, w: int, h: int) -> CameraModel:
        key = (w, h)
        if key not in self._cams:
            self._cams[key] = CameraModel(w, h, self.hfov)
        return self._cams[key]

    @torch.no_grad()
    def process(self, frame: np.ndarray, timestamp: float | None = None) -> FrameResult:
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be an HxWx3 BGR image")
        tm: dict[str, float] = {}
        t_start = t = time.perf_counter()

        h0, w0 = frame.shape[:2]
        if max(h0, w0) > self.max_side:
            s = self.max_side / max(h0, w0)
            frame = cv2.resize(frame, (int(w0 * s), int(h0 * s)), interpolation=cv2.INTER_AREA)
        blob, meta = self.detector.preprocess(frame)
        tm["preprocess"] = (now := time.perf_counter()) - t; t = now

        # infer() copies results to the host, which syncs the GPU, so this timing is real.
        cand = self.detector.infer(blob)
        tm["detect"] = (now := time.perf_counter()) - t; t = now

        dets = self.detector.postprocess(cand, meta)
        if not self.announce_all:
            dets = [d for d in dets if d.label in OBSTACLE_CLASSES]
        tm["nms"] = (now := time.perf_counter()) - t; t = now

        postures: dict[int, tuple[str, float]] = {}
        persons = sorted((i for i, d in enumerate(dets) if d.label == "person"),
                         key=lambda i: -dets[i].box[3])[: self.max_crops]
        if self.posture is not None and persons:
            preds = self.posture.predict(frame, [dets[i].box for i in persons])
            postures = dict(zip(persons, preds))
        tm["posture"] = (now := time.perf_counter()) - t; t = now

        cam = self._camera(frame.shape[1], frame.shape[0])
        objs: list[ObjectReport] = []
        for i, d in enumerate(dets):
            p_lbl, p_conf = postures.get(i, (None, None))
            height_posture = p_lbl if self.say_posture else None
            sp = localize(d.box, d.label, cam, posture=height_posture, clock_mode=self.clock_mode)
            objs.append(ObjectReport(d.label, d.confidence, d.box, sp, p_lbl, p_conf))
        if timestamp is not None:
            self.tracker.update(objs, timestamp)
        instruction, announcements, action, urgency = self.planner.plan(objs, self.say_posture)
        tm["localize_plan"] = (now := time.perf_counter()) - t

        tm["total"] = now - t_start
        return FrameResult(instruction, announcements, objs, action, urgency,
                           {k: v * 1000 for k, v in tm.items()}, self.budget_ms,
                           (frame.shape[1], frame.shape[0]))


def annotate(frame: np.ndarray, res: FrameResult) -> np.ndarray:
    out = frame.copy()
    if (frame.shape[1], frame.shape[0]) != res.image_size:
        out = cv2.resize(out, res.image_size)
    for o in res.objects:
        x, y, w, h = o.box
        col = (0, 0, 255) if o.in_path else (0, 200, 0)
        cv2.rectangle(out, (x, y), (x + w, y + h), col, 2)
        d = o.spatial.distance_m
        txt = f"{o.label}{'/' + o.posture if o.posture else ''} {o.spatial.clock_hour}h"
        txt += f" {d:.1f}m" if d is not None else ""
        cv2.putText(out, txt, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, out.shape[0] - 22), (out.shape[1], out.shape[0]), (0, 0, 0), -1)
    cv2.putText(out, f"{res.instruction}  [{res.total_ms:.0f} ms]", (6, out.shape[0] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


class Speaker:
    """Non-blocking TTS via macOS `say` / Linux `espeak`; never stalls the pipeline."""

    def __init__(self, min_interval_s: float = 2.5):
        self.cmd = "say" if shutil.which("say") else ("espeak" if shutil.which("espeak") else None)
        if self.cmd is None:
            log.warning("no TTS command found (say/espeak); --speak disabled")
        self.proc: subprocess.Popen | None = None
        self.last_text, self.last_t, self.min_int = "", 0.0, min_interval_s

    def say(self, text: str) -> None:
        if self.cmd is None:
            return
        now = time.time()
        if text == self.last_text and now - self.last_t < self.min_int * 2:
            return
        if self.proc is not None and self.proc.poll() is None:
            if not text.startswith("Stop."):
                return
            self.proc.terminate()
        self.proc = subprocess.Popen([self.cmd, text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.last_text, self.last_t = text, now


def _stats(vals: list[float]) -> tuple[float, float, float, float]:
    s = sorted(vals)
    p95 = s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]
    return statistics.mean(s), statistics.median(s), p95, s[-1]


def print_report(pipe: NavigationPipeline, results: list[FrameResult]) -> bool:
    last = results[-1]
    print("\n=== Scene ===")
    for o in sorted(last.objects, key=lambda o: o.spatial.distance_m or 1e9):
        d = o.spatial.distance_m
        post = f" posture={o.posture}({o.posture_conf:.2f}){'' if pipe.say_posture else '[untrained]'}" \
            if o.posture else ""
        print(f"  {o.label:<13} conf={o.confidence:.2f} box={o.box} bearing={o.spatial.bearing_deg:+5.1f}deg "
              f"{o.spatial.clock_str:<11} dist={'%.2fm' % d if d else 'n/a':<6}"
              f"{' (upper bound)' if o.spatial.distance_is_upper_bound else ''}"
              f"{' IN-PATH' if o.in_path else ''}{post}")
    print("\n=== Instruction ===")
    print(f"  >> {last.instruction}")
    for a in last.announcements:
        print(f"     - {a}")

    print(f"\n=== Latency ({len(results)} warm run(s), device={pipe.device}, model={pipe.model_name}, "
          f"input={pipe.detector.input_size}) ===")
    cold = f"   cold first frame (warm-up): {pipe.cold_ms:.1f} ms" if pipe.cold_ms is not None else ""
    print(f"  model load: {pipe.load_s * 1000:.0f} ms{cold}")
    print(f"  {'stage':<14}{'mean':>9}{'p50':>9}{'p95':>9}{'max':>9}  (ms)")
    for st in (*STAGES, "total"):
        m, p50, p95, mx = _stats([r.timings_ms[st] for r in results])
        print(f"  {st:<14}{m:9.2f}{p50:9.2f}{p95:9.2f}{mx:9.2f}")
    _, _, p95, mx = _stats([r.total_ms for r in results])
    ok = p95 < pipe.budget_ms
    print(f"\n  Budget {pipe.budget_ms:.0f} ms -> p95 total {p95:.1f} ms, max {mx:.1f} ms : "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


FLIP_CODES = {"none": None, "mirror": 1, "vertical": 0, "rotate180": -1}


def flip_frame(frame: np.ndarray, mode: str) -> np.ndarray:
    """Flip before processing so directions and the display match what is shown."""
    code = FLIP_CODES[mode]
    return frame if code is None else cv2.flip(frame, code)


BLACK_LEVEL = 8            # max pixel value below which a frame counts as black
BLACK_ALERT_FRAMES = 15    # ~0.5 s at 30 fps before announcing the feed is lost
BLACK_REOPEN_FRAMES = 90   # ~3 s of black frames -> release and reopen the camera


def is_black(frame: np.ndarray) -> bool:
    return int(frame.max()) < BLACK_LEVEL


def run_stream(pipe: NavigationPipeline, source, args) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video source {source!r}")
    speaker = Speaker() if args.speak else None
    writer = None
    can_show = args.show
    smoother = InstructionSmoother()
    lat: list[float] = []
    live = not isinstance(source, str)
    frames, failed_reads, black_frames = 0, 0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # Cameras (especially macOS Continuity/iPhone cameras) can open yet
                # deliver no frames for a moment; retry before giving up.
                failed_reads += 1
                if live and failed_reads < 50:
                    time.sleep(0.05)
                    continue
                if frames == 0:
                    hint = (" Try another index, e.g. --camera 1 (on macOS, index 0 is often"
                            " an iPhone Continuity Camera), and check camera permission for"
                            " your terminal app." if live else "")
                    raise SystemExit(f"no frames received from video source {source!r}.{hint}")
                break
            failed_reads = 0
            frames += 1
            frame = flip_frame(frame, args.flip)
            if live:
                # A black frame contains no obstacles, so running it through the
                # planner would report a clear path while the camera is blind.
                # Skip it, tell the user, and reopen the camera if it stays dark
                # (an iPhone Continuity Camera goes black when the phone is
                # unlocked, moved, covered or out of range).
                if is_black(frame):
                    black_frames += 1
                    if black_frames == BLACK_ALERT_FRAMES:
                        msg = "Stop. Camera feed lost."
                        log.warning("camera %r is delivering black frames - check that the iPhone is "
                                    "locked, still and uncovered, or use --camera 1 for the built-in "
                                    "camera", source)
                        print(f"[camera] {msg}")
                        if speaker:
                            speaker.say(msg)
                    if black_frames % BLACK_REOPEN_FRAMES == 0:
                        log.info("reopening camera %r", source)
                        cap.release()
                        cap = cv2.VideoCapture(source)
                    continue
                if black_frames >= BLACK_ALERT_FRAMES:
                    print("[camera] Camera feed restored.")
                    if speaker:
                        speaker.say("Camera feed restored.")
                black_frames = 0
            # Files decode faster than real time, so use their own timestamps for
            # the approach-speed estimate; live cameras use the wall clock.
            if isinstance(source, str):
                ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            else:
                ts = time.perf_counter()
            res = pipe.process(frame, timestamp=ts)
            lat.append(res.total_ms)
            spoken = smoother.update(res, ts)
            if spoken:
                print(f"[t={ts:6.2f}s {res.total_ms:6.1f} ms] {spoken}")
                if speaker:
                    speaker.say(spoken)
            if args.save or can_show:
                vis = annotate(frame, res)
                if args.save:
                    if writer is None:
                        fps = cap.get(cv2.CAP_PROP_FPS) or 15
                        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                                 (vis.shape[1], vis.shape[0]))
                    writer.write(vis)
                if can_show:
                    try:
                        cv2.imshow("navigation", vis)
                        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                            break
                    except cv2.error:
                        log.warning("OpenCV built without GUI support (headless) - disabling --show")
                        can_show = False
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if can_show:
            cv2.destroyAllWindows()
    if lat:
        m, p50, p95, mx = _stats(lat)
        print(f"\n{len(lat)} frames: mean {m:.1f} ms, p50 {p50:.1f}, p95 {p95:.1f}, max {mx:.1f} "
              f"-> {'PASS' if p95 < pipe.budget_ms else 'FAIL'} ({pipe.budget_ms:.0f} ms budget)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Assistive navigation pipeline for blind / low-vision users")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="path to an image")
    src.add_argument("--video", help="path to a video file")
    src.add_argument("--camera", type=int, help="webcam index")
    ap.add_argument("--weights-dir", default=str(ROOT / "weights"))
    ap.add_argument("--model", default="auto", choices=["auto", "yolov3", "yolov3-tiny"])
    ap.add_argument("--input-size", type=int, default=416, help="YOLO input (multiple of 32)")
    ap.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    ap.add_argument("--half", action="store_true", help="FP16 YOLO on mps/cuda")
    ap.add_argument("--hfov", type=float, default=60.0, help="camera horizontal FOV in degrees")
    ap.add_argument("--flip", default="none", choices=list(FLIP_CODES),
                    help="flip input frames: mirror (left-right), vertical (upside down), rotate180")
    ap.add_argument("--clock-mode", default="angular", choices=["angular", "fov"])
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--posture-checkpoint", default=None)
    ap.add_argument("--announce-all", action="store_true", help="include non-obstacle classes")
    ap.add_argument("--runs", type=int, default=10, help="timed repetitions for --image")
    ap.add_argument("--budget-ms", type=float, default=300.0)
    ap.add_argument("--save", help="write annotated image / video here")
    ap.add_argument("--json", action="store_true", help="print the final FrameResult as JSON")
    ap.add_argument("--speak", action="store_true", help="speak instructions (say/espeak)")
    ap.add_argument("--show", action="store_true", help="display annotated stream")
    ap.add_argument("--strict", action="store_true", help="exit 1 if the latency budget is missed")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    pipe = NavigationPipeline(args.weights_dir, args.model, args.input_size, args.device, args.hfov,
                              args.clock_mode, args.conf, posture_checkpoint=args.posture_checkpoint,
                              budget_ms=args.budget_ms, announce_all=args.announce_all, half=args.half)

    if args.image is None:
        run_stream(pipe, args.video if args.video else args.camera, args)
        return 0

    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"cannot read image {args.image!r}")
    frame = flip_frame(frame, args.flip)
    results = [pipe.process(frame) for _ in range(max(1, args.runs))]
    ok = print_report(pipe, results)
    if args.save:
        cv2.imwrite(args.save, annotate(frame, results[-1]))
        print(f"  annotated image -> {args.save}")
    if args.json:
        print(json.dumps(results[-1].to_dict(), indent=2, default=float))
    if args.speak:
        Speaker().say(results[-1].instruction)
    return 0 if (ok or not args.strict) else 1


if __name__ == "__main__":
    sys.exit(main())
