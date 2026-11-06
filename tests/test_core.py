"""Unit tests that need no downloaded weights (run with: pytest -q)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_posture_dataset import label_from_geometry, posture_features  # noqa: E402
from geometry import (CameraModel, clock_direction, estimate_distance, format_distance,  # noqa: E402
                      localize)
from models import Darknet, letterbox  # noqa: E402
from pipeline import FrameResult, InstructionSmoother, NavigationPlanner, ObjectReport  # noqa: E402

CAM = CameraModel(640, 480, 60.0)


def test_focal_length_matches_pinhole():
    assert CAM.focal_px == pytest.approx(320 / math.tan(math.radians(30)))


def test_distance_inverts_projection():
    h_px = 1.7 * CAM.focal_px / 2.0  # 1.7 m person at 2 m
    assert estimate_distance([300, 50, 40, h_px], CAM, 1.7) == pytest.approx(2.0)


@pytest.mark.parametrize("cx,angular,fov", [(0, 11, 9), (320, 12, 12), (640, 1, 3), (440, 12, 1)])
def test_clock_modes(cx, angular, fov):
    box = [cx - 5, 100, 10, 100]
    assert clock_direction(box, CAM, "angular")[0] == angular
    assert clock_direction(box, CAM, "fov")[0] == fov


def test_wide_fov_clamped_to_9_to_3():
    wide = CameraModel(640, 480, 170.0)
    assert clock_direction([0, 0, 2, 10], wide, "angular")[0] == 9
    assert clock_direction([638, 0, 2, 10], wide, "angular")[0] == 3


def test_sitting_person_uses_shorter_height():
    box = [300, 100, 40, 200]
    stand = localize(box, "person", CAM, posture="standing").distance_m
    sit = localize(box, "person", CAM, posture="sitting").distance_m
    assert sit == pytest.approx(stand * 1.3 / 1.7)


def test_truncated_box_is_upper_bound_and_unknown_class():
    assert localize([300, 0, 40, 200], "chair", CAM).distance_is_upper_bound
    assert localize([300, 50, 40, 200], "kite", CAM).distance_m is None
    assert format_distance(3.0, True) == "within 3 meters"
    assert format_distance(0.4) == "less than 1 meter"


def _obj(label, box, cam=CAM):
    return ObjectReport(label, 0.9, tuple(box), localize(box, label, cam))


def _box_at(distance_m, center_x, width_px=80, label_h=0.9):
    h = label_h * CAM.focal_px / distance_m
    return [center_x - width_px // 2, 240 - h / 2, width_px, h]


def test_clear_path():
    instr, _, action, urg = NavigationPlanner().plan([], say_posture=False)
    assert action == "clear" and urg == 0 and "clear" in instr.lower()


def test_close_obstacle_ahead_stops_and_offers_side():
    chair = _obj("chair", _box_at(0.8, 320))
    instr, _, action, urg = NavigationPlanner().plan([chair], say_posture=False)
    assert urg == 3 and instr.startswith("Stop.")
    assert action in ("left", "right")


def test_obstacle_offset_left_suggests_right():
    chair = _obj("chair", _box_at(1.5, 270))
    _, _, action, urg = NavigationPlanner().plan([chair], say_posture=False)
    assert urg == 2 and action == "right"


def test_both_sides_blocked():
    objs = [_obj("chair", _box_at(1.5, x, width_px=200)) for x in (120, 320, 520)]
    _, _, action, _ = NavigationPlanner().plan(objs, say_posture=False)
    assert action == "blocked"


def _res(action, urgency):
    return FrameResult(f"{action}/{urgency}", [], [], action, urgency, {"total": 1.0}, 300, (640, 480))


def test_smoother_hysteresis_and_stop_escalation():
    sm = InstructionSmoother(persist=3, repeat_s=100)
    assert sm.update(_res("clear", 0), 0.0) == "clear/0"        # first state announced
    assert sm.update(_res("left", 2), 0.1) is None              # needs 3 frames
    assert sm.update(_res("right", 2), 0.2) is None             # flicker resets
    assert sm.update(_res("right", 2), 0.3) is None
    assert sm.update(_res("right", 2), 0.4) == "right/2"        # persisted
    assert sm.update(_res("blocked", 3), 0.5) == "blocked/3"    # Stop is immediate


TINY_CFG = """[net]
width=64
height=64
channels=3

[convolutional]
batch_normalize=1
filters=8
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=8
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-2
activation=linear

[convolutional]
size=1
stride=1
pad=1
filters=21
activation=linear

[yolo]
mask=0,1,2
anchors=10,14, 23,27, 37,58
classes=2
num=3

[route]
layers=-3

[upsample]
stride=2

[maxpool]
size=2
stride=1

[convolutional]
size=1
stride=1
pad=1
filters=21
activation=linear

[yolo]
mask=0,1,2
anchors=10,14, 23,27, 37,58
classes=2
num=3
"""


def test_darknet_cfg_and_weights_roundtrip(tmp_path):
    cfg = tmp_path / "tiny.cfg"
    cfg.write_text(TINY_CFG)
    net = Darknet(cfg)
    n_params = sum(m.conv.weight.numel() + (m.bn.weight.numel() * 4 if "bn" in m._modules else m.conv.bias.numel())
                   for b, m in zip(net.blocks, net.module_list) if b["type"] == "convolutional")
    rng = np.random.default_rng(0)
    w = rng.standard_normal(n_params).astype(np.float32)
    wf = tmp_path / "tiny.weights"
    with open(wf, "wb") as fh:
        np.array([0, 2, 0], np.int32).tofile(fh)
        np.array([0], np.int64).tofile(fh)
        w.tofile(fh)
    net.load_darknet_weights(wf)
    first = net.module_list[0]
    assert torch.allclose(first.bn.bias, torch.from_numpy(w[:8]))
    out = net.eval()(torch.zeros(1, 3, 64, 64))
    # stride-2 head: 32x32 grid, upsampled head: 64x64 grid; 3 anchors each, 5+2 values
    assert out.shape == (1, 3 * 32 * 32 + 3 * 64 * 64, 7)

    with open(wf, "ab") as fh:  # extra weights -> mismatch detected
        np.zeros(5, np.float32).tofile(fh)
    with pytest.raises(ValueError):
        Darknet(cfg).load_darknet_weights(wf)


def test_letterbox_preserves_aspect():
    img = np.zeros((100, 200, 3), np.uint8)
    out, scale, (px, py) = letterbox(img, 64)
    assert out.shape == (64, 64, 3) and scale == pytest.approx(0.32) and px == 0 and py == 16


def _person(knee_dx=0, knee_y=200, ank=(0, 300), stride=10):
    return {"l_sh": (90, 0), "r_sh": (110, 0), "l_hip": (92, 100), "r_hip": (108, 100),
            "l_knee": (92 + knee_dx, knee_y), "r_knee": (108 + knee_dx, knee_y),
            "l_ank": (92 + ank[0] - stride / 2, ank[1]), "r_ank": (108 + ank[0] + stride / 2, ank[1])}


def test_standing_geometry():
    assert label_from_geometry(posture_features(_person()), "coco", "a man standing") == "standing"


def test_walking_needs_caption():
    f = posture_features(_person(stride=60, ank=(0, 290)))
    f["knee_min"] = 160.0  # slight swing-leg bend
    assert label_from_geometry(f, "coco", "a man walking down the street") == "walking"
    assert label_from_geometry(f, "coco", "a man playing tennis") is None


def test_sitting_needs_seat_or_caption():
    f = posture_features(_person(knee_dx=90, knee_y=110, ank=(90, 210)))
    assert label_from_geometry(f, "coco", "", seat_under_hips=True) == "sitting"
    assert label_from_geometry(f, "coco", "a woman sitting on a bench") == "sitting"
    assert label_from_geometry(f, "coco", "a woman on a skateboard") is None
    assert label_from_geometry(f, "coco", "sitting", sport_overlap=True) is None


def test_live_stream_skips_black_frames(monkeypatch, capsys):
    """Black frames must never reach the planner: an empty frame would read as a clear path."""
    import types

    import pipeline as pl

    black = np.zeros((48, 64, 3), np.uint8)
    real = np.full((48, 64, 3), 120, np.uint8)
    seq = [black] * pl.BLACK_ALERT_FRAMES + [real, real]

    class FakeCap:
        def __init__(self, *_):
            self.i = 0

        def isOpened(self):
            return True

        def read(self):
            if self.i >= len(seq):
                raise KeyboardInterrupt  # end the live loop
            self.i += 1
            return True, seq[self.i - 1]

        def get(self, _):
            return 0.0

        def release(self):
            pass

    seen = []

    class FakePipe:
        budget_ms = 100.0

        def process(self, frame, timestamp=None):
            seen.append(int(frame.max()))
            return FrameResult(instruction="", announcements=[], objects=[], action="clear",
                               urgency=0, timings_ms={"total": 1.0}, budget_ms=100.0,
                               image_size=(64, 48))

    monkeypatch.setattr(pl.cv2, "VideoCapture", FakeCap)
    args = types.SimpleNamespace(speak=False, show=False, save=None, flip="none")
    pl.run_stream(FakePipe(), 0, args)
    out = capsys.readouterr().out
    assert seen == [120, 120]
    assert "Camera feed lost" in out and "Camera feed restored" in out
