"""Spatial localisation: 2-D bounding boxes -> clock direction + metric distance.

Conventions
-----------
* Boxes are ``[x, y, w, h]`` in pixels, (x, y) = top-left corner.
* Camera is modelled as an ideal pinhole with square pixels and the principal
  point at the image centre. The horizontal field of view (HFOV, default 60 deg)
  gives the focal length in pixels::

        f_px = (W / 2) / tan(HFOV / 2)

* Bearing is the horizontal angle between the optical axis and the ray through
  the box centre (negative = left, positive = right)::

        bearing = atan((cx - W/2) / f_px)

* Distance uses similar triangles with a known real-world object height::

        Z = H_real * f_px / h_px

Clock directions
----------------
12 o'clock is straight ahead; each clock hour is 30 deg. Two mappings are
offered and both are clamped to the 9 o'clock to 3 o'clock frontal range:

* ``"angular"`` (default) - physically correct: hour = 12 + round(bearing / 30).
  With a 60 deg camera the image edges are at +/-30 deg, i.e. 11 to 1 o'clock.
  Use this when the user physically turns toward the announced hour.
* ``"fov"`` - linear: the full camera HFOV is spread over 9 to 3 o'clock
  (left edge -> 9, centre -> 12, right edge -> 3). Finer-grained relative
  layout, but the hour no longer equals a true heading.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

# Typical real-world heights (metres) for COCO classes found indoors / on
# pedestrian routes. Values are averages for the upright object as usually
# photographed; used by the pinhole distance estimate.
KNOWN_HEIGHTS_M: dict[str, float] = {
    "person": 1.70,
    "chair": 0.90,
    "couch": 0.85,
    "bed": 0.60,
    "dining table": 0.75,
    "toilet": 0.75,
    "tv": 0.60,
    "laptop": 0.25,
    "refrigerator": 1.75,
    "oven": 0.90,
    "microwave": 0.30,
    "sink": 0.90,
    "potted plant": 0.60,
    "bench": 0.85,
    "suitcase": 0.65,
    "backpack": 0.50,
    "handbag": 0.30,
    "bottle": 0.25,
    "cup": 0.10,
    "vase": 0.30,
    "book": 0.25,
    "clock": 0.30,
    "dog": 0.55,
    "cat": 0.30,
    "bicycle": 1.05,
    "motorcycle": 1.10,
    "car": 1.50,
    "bus": 3.20,
    "truck": 3.00,
    "fire hydrant": 0.75,
    "stop sign": 2.10,
    "parking meter": 1.30,
    "traffic light": 1.00,
    "umbrella": 1.00,
    "keyboard": 0.04,
    "mouse": 0.04,
    "remote": 0.05,
    "cell phone": 0.15,
}

# Person height depends on posture: a seated adult is ~1.30 m tall.
POSTURE_HEIGHTS_M: dict[str, float] = {
    "standing": 1.70,
    "walking": 1.70,
    "sitting": 1.30,
}

_EDGE_TOL_PX = 2  # box within this many px of the border counts as truncated


@dataclass(frozen=True)
class CameraModel:
    """Ideal pinhole camera. ``hfov_deg`` is the horizontal field of view."""

    image_width: int
    image_height: int
    hfov_deg: float = 60.0

    def __post_init__(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive")
        if not 1.0 <= self.hfov_deg < 179.0:
            raise ValueError("hfov_deg must be in [1, 179)")

    @property
    def focal_px(self) -> float:
        return (self.image_width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan((self.image_height / 2.0) / self.focal_px))

    @property
    def cx(self) -> float:
        return self.image_width / 2.0

    @property
    def cy(self) -> float:
        return self.image_height / 2.0


@dataclass
class SpatialInfo:
    bearing_deg: float               # + right / - left of optical axis
    clock_hour: int                  # 9, 10, 11, 12, 1, 2, 3
    clock_str: str                   # "12 o'clock"
    distance_m: float | None         # None if object height unknown
    distance_is_upper_bound: bool    # box clipped by frame top/bottom
    lateral_m: tuple[float, float] | None  # (left, right) edge offsets at distance_m
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def bearing_deg(x_px: float, cam: CameraModel) -> float:
    """Horizontal angle (deg) of image column ``x_px`` relative to the optical axis."""
    return math.degrees(math.atan((x_px - cam.cx) / cam.focal_px))


def _hour_from_offset(offset_hours: int) -> int:
    offset_hours = max(-3, min(3, offset_hours))
    return (12 + offset_hours - 1) % 12 + 1


def clock_direction(box: Sequence[float], cam: CameraModel, mode: str = "angular") -> tuple[int, float]:
    """Return ``(clock_hour, bearing_deg)`` for a box ``[x, y, w, h]``.

    ``clock_hour`` is in {9, 10, 11, 12, 1, 2, 3}.
    """
    x, _, w, _ = box
    cx = x + w / 2.0
    bear = bearing_deg(cx, cam)
    if mode == "angular":
        offset = round(bear / 30.0)
    elif mode == "fov":
        # Normalised horizontal position in [-1, 1] spread over +/-3 hours.
        norm = (cx - cam.cx) / (cam.image_width / 2.0)
        offset = round(max(-1.0, min(1.0, norm)) * 3.0)
    else:
        raise ValueError(f"unknown clock mode {mode!r} (use 'angular' or 'fov')")
    return _hour_from_offset(int(offset)), bear


def clock_to_str(hour: int) -> str:
    return f"{hour} o'clock"


def estimate_distance(
    box: Sequence[float],
    cam: CameraModel,
    real_height_m: float,
) -> float:
    """Pinhole-model range (metres) along the optical axis: Z = H * f / h."""
    h_px = float(box[3])
    if h_px <= 0:
        raise ValueError("box height must be positive")
    if real_height_m <= 0:
        raise ValueError("real_height_m must be positive")
    return real_height_m * cam.focal_px / h_px


def object_height(label: str, posture: str | None = None,
                  overrides: dict[str, float] | None = None) -> float | None:
    if overrides and label in overrides:
        return overrides[label]
    if label == "person" and posture in POSTURE_HEIGHTS_M:
        return POSTURE_HEIGHTS_M[posture]
    return KNOWN_HEIGHTS_M.get(label)


def is_vertically_truncated(box: Sequence[float], cam: CameraModel) -> bool:
    _, y, _, h = box
    return y <= _EDGE_TOL_PX or (y + h) >= cam.image_height - _EDGE_TOL_PX


def lateral_extent(box: Sequence[float], cam: CameraModel, distance_m: float) -> tuple[float, float]:
    """Left/right box edges as sideways offsets (metres) at ``distance_m``."""
    x, _, w, _ = box
    left = (x - cam.cx) * distance_m / cam.focal_px
    right = (x + w - cam.cx) * distance_m / cam.focal_px
    return left, right


def localize(
    box: Sequence[float],
    label: str,
    cam: CameraModel,
    posture: str | None = None,
    clock_mode: str = "angular",
    height_overrides: dict[str, float] | None = None,
) -> SpatialInfo:
    """Full spatial description of one detection."""
    hour, bear = clock_direction(box, cam, clock_mode)
    real_h = object_height(label, posture, height_overrides)
    truncated = is_vertically_truncated(box, cam)
    dist = lat = None
    if real_h is not None and box[3] > 0:
        dist = estimate_distance(box, cam, real_h)
        lat = lateral_extent(box, cam, dist)
    return SpatialInfo(
        bearing_deg=bear,
        clock_hour=hour,
        clock_str=clock_to_str(hour),
        distance_m=dist,
        distance_is_upper_bound=truncated and dist is not None,
        lateral_m=lat,
    )


def localize_many(
    boxes: Iterable[Sequence[float]],
    labels: Iterable[str],
    cam: CameraModel,
    postures: Iterable[str | None] | None = None,
    clock_mode: str = "angular",
) -> list[SpatialInfo]:
    boxes, labels = list(boxes), list(labels)
    postures = list(postures) if postures is not None else [None] * len(boxes)
    return [localize(b, l, cam, p, clock_mode) for b, l, p in zip(boxes, labels, postures)]


def format_distance(d: float | None, upper_bound: bool = False) -> str:
    if d is None:
        return "distance unknown"
    if d < 1.0:
        txt = "less than 1 meter" if not upper_bound else "very close"
        return txt
    val = round(d * 2) / 2  # nearest half metre for speech
    unit = "meter" if val == 1 else "meters"
    num = f"{val:g}"
    return f"within {num} {unit}" if upper_bound else f"{num} {unit}"


if __name__ == "__main__":
    cam = CameraModel(640, 480, 60.0)
    print(f"focal = {cam.focal_px:.1f}px, vfov = {cam.vfov_deg:.1f} deg")
    for cx in (0, 80, 200, 320, 440, 560, 640):
        box = [cx - 10, 100, 20, 200]
        a, _ = clock_direction(box, cam, "angular")
        f, b = clock_direction(box, cam, "fov")
        print(f"cx={cx:3d} bearing={b:+6.1f}  angular={a:2d}  fov={f:2d}")
    s = localize([280, 60, 80, 300], "person", cam)
    print(s)
