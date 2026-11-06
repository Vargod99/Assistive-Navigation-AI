#!/usr/bin/env python3
"""Download YOLOv3 assets and initialise the posture-classifier checkpoint.

Creates (inside ``weights/``):
    coco.names                 80 COCO class labels
    yolov3.cfg / .weights      full YOLOv3 (Darknet format, ~248 MB)
    yolov3-tiny.cfg / .weights YOLOv3-tiny fast fallback (~34 MB)
    mobilenetv2_posture.pth    MobileNetV2 posture model (ImageNet backbone +
                               freshly-initialised 3-class head) so the pipeline
                               runs out of the box. Replace it by running
                               ``train_posture.py`` on real data.
And ``test_scene.jpg`` in the project root (a COCO indoor photo with a person
and chairs; a synthetic scene is rendered instead if offline).

Usage:
    python setup_weights.py                 # everything
    python setup_weights.py --model tiny    # only YOLOv3-tiny (+ names, posture)
    python setup_weights.py --force         # re-download / re-create
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

_DARKNET_RAW = "https://raw.githubusercontent.com/pjreddie/darknet/master"

# name -> (list of mirror URLs, expected size in bytes or None)
ASSETS: dict[str, tuple[list[str], int | None]] = {
    "coco.names": ([f"{_DARKNET_RAW}/data/coco.names"], None),
    "yolov3.cfg": ([f"{_DARKNET_RAW}/cfg/yolov3.cfg"], None),
    "yolov3-tiny.cfg": ([f"{_DARKNET_RAW}/cfg/yolov3-tiny.cfg"], None),
    "yolov3.weights": (
        [
            "https://pjreddie.com/media/files/yolov3.weights",
            "https://data.pjreddie.com/files/yolov3.weights",
        ],
        248_007_048,
    ),
    "yolov3-tiny.weights": (
        [
            "https://pjreddie.com/media/files/yolov3-tiny.weights",
            "https://data.pjreddie.com/files/yolov3-tiny.weights",
        ],
        35_434_956,
    ),
}

MODEL_ASSETS = {
    "full": ["coco.names", "yolov3.cfg", "yolov3.weights"],
    "tiny": ["coco.names", "yolov3-tiny.cfg", "yolov3-tiny.weights"],
}

POSTURE_CKPT = "mobilenetv2_posture.pth"

# COCO val2017 #139: indoor living room with a person, several chairs, a TV,
# a dining table and potted plants (COCO images are CC-BY licensed).
TEST_IMAGE_URL = "http://images.cocodataset.org/val2017/000000000139.jpg"
TEST_IMAGE = Path(__file__).resolve().parent / "test_scene.jpg"


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def _download(url: str, dest: Path, expected_size: int | None, retries: int = 3) -> None:
    """Stream ``url`` to ``dest`` atomically (via a .part file) with progress."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "blind-nav-setup/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as fh:
                total = int(resp.headers.get("Content-Length") or 0) or expected_size or 0
                done, t0, last_print = 0, time.time(), 0.0
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    now = time.time()
                    if sys.stdout.isatty() and now - last_print > 0.25:
                        rate = done / max(now - t0, 1e-6)
                        pct = f"{100 * done / total:5.1f}%" if total else "  ?  "
                        print(f"\r    {pct} {_human(done)} @ {_human(rate)}/s", end="", flush=True)
                        last_print = now
                if sys.stdout.isatty():
                    print()
            size = tmp.stat().st_size
            if expected_size is not None and size != expected_size:
                raise IOError(f"size mismatch: got {size}, expected {expected_size}")
            if size == 0:
                raise IOError("empty download")
            os.replace(tmp, dest)
            return
        except (urllib.error.URLError, IOError, TimeoutError) as exc:
            last_err = exc
            print(f"    attempt {attempt}/{retries} failed: {exc}")
            time.sleep(min(2 ** attempt, 10))
    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"could not download {url}: {last_err}")


def fetch_asset(name: str, force: bool = False) -> Path:
    urls, expected = ASSETS[name]
    dest = WEIGHTS_DIR / name
    if dest.exists() and not force:
        size = dest.stat().st_size
        if (expected is None and size > 0) or size == expected:
            print(f"[ok]   {name} already present ({_human(size)})")
            return dest
        print(f"[warn] {name} looks incomplete ({size} bytes) - re-downloading")
    errors = []
    for url in urls:
        print(f"[get]  {name} <- {url}")
        try:
            _download(url, dest, expected)
            print(f"[ok]   {name} ({_human(dest.stat().st_size)})")
            return dest
        except RuntimeError as exc:
            errors.append(str(exc))
    raise RuntimeError(f"all mirrors failed for {name}:\n  " + "\n  ".join(errors))


def init_posture_checkpoint(force: bool = False) -> Path:
    """Create a MobileNetV2 posture checkpoint with an ImageNet backbone.

    The 3-class head is randomly initialised (seeded), and the checkpoint is
    marked ``trained=False`` so the pipeline knows not to announce its posture
    labels to the user until the model has been fine-tuned.
    """
    import torch

    from models import POSTURE_CLASSES, build_posture_model

    dest = WEIGHTS_DIR / POSTURE_CKPT
    if dest.exists() and not force:
        try:
            ckpt = torch.load(dest, map_location="cpu", weights_only=False)
            state = "fine-tuned" if ckpt.get("trained") else "untrained head"
            print(f"[ok]   {POSTURE_CKPT} already present ({state})")
            return dest
        except Exception as exc:  # corrupt file -> rebuild
            print(f"[warn] {POSTURE_CKPT} unreadable ({exc}); re-creating")

    torch.manual_seed(0)
    backbone = "imagenet"
    try:
        model = build_posture_model(num_classes=len(POSTURE_CLASSES), pretrained=True)
    except Exception as exc:
        print(f"[warn] could not fetch ImageNet weights ({exc}); using random init")
        model = build_posture_model(num_classes=len(POSTURE_CLASSES), pretrained=False)
        backbone = "random"

    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": list(POSTURE_CLASSES),
            "img_size": 224,
            "arch": "mobilenet_v2",
            "trained": False,
            "backbone_init": backbone,
            "val_acc": None,
        },
        dest,
    )
    print(f"[ok]   {POSTURE_CKPT} initialised (backbone={backbone}, head untrained)")
    return dest


def make_synthetic_scene(dest: Path) -> None:
    """Offline fallback: a rendered room with a stick-figure person and a chair.

    YOLOv3 may not detect drawn shapes; this only guarantees the pipeline has
    an input to run (and time) when the sample photo cannot be downloaded.
    """
    import cv2
    import numpy as np

    h, w = 480, 640
    img = np.zeros((h, w, 3), np.uint8)
    img[:300] = (205, 215, 225)                                  # wall
    img[300:] = (95, 120, 150)                                   # floor
    for x in range(0, w, 80):                                    # floor boards
        cv2.line(img, (x, 300), (int(w / 2 + (x - w / 2) * 2.2), h), (80, 100, 130), 1)
    cv2.rectangle(img, (470, 120), (560, 300), (60, 80, 120), -1)  # door
    # chair at 11 o'clock
    c = (40, 60, 110)
    cv2.rectangle(img, (150, 180), (160, 330), c, -1)
    cv2.rectangle(img, (150, 260), (240, 275), c, -1)
    for x in (152, 230):
        cv2.rectangle(img, (x, 275), (x + 8, 340), c, -1)
    # person at 1 o'clock
    p = (60, 50, 40)
    cv2.circle(img, (400, 150), 22, (150, 180, 220), -1)
    cv2.rectangle(img, (378, 175), (422, 290), (170, 80, 40), -1)
    cv2.rectangle(img, (380, 290), (398, 400), p, -1)
    cv2.rectangle(img, (402, 290), (420, 400), p, -1)
    cv2.rectangle(img, (362, 180), (376, 280), (170, 80, 40), -1)
    cv2.rectangle(img, (424, 180), (438, 280), (170, 80, 40), -1)
    cv2.imwrite(str(dest), img)


def fetch_test_image(force: bool = False) -> Path:
    if TEST_IMAGE.exists() and not force:
        print(f"[ok]   {TEST_IMAGE.name} already present")
        return TEST_IMAGE
    try:
        print(f"[get]  {TEST_IMAGE.name} <- {TEST_IMAGE_URL}")
        _download(TEST_IMAGE_URL, TEST_IMAGE, None)
        print(f"[ok]   {TEST_IMAGE.name} (COCO val2017 #139: person, chairs, TV, table)")
    except RuntimeError as exc:
        print(f"[warn] {exc}\n       generating a synthetic scene instead")
        make_synthetic_scene(TEST_IMAGE)
        print(f"[ok]   {TEST_IMAGE.name} (synthetic)")
    return TEST_IMAGE


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["all", "full", "tiny"], default="all",
                    help="which YOLOv3 variant(s) to download (default: all)")
    ap.add_argument("--skip-posture", action="store_true", help="don't create the posture checkpoint")
    ap.add_argument("--skip-test-image", action="store_true", help="don't fetch test_scene.jpg")
    ap.add_argument("--force", action="store_true", help="re-download / re-create existing files")
    args = ap.parse_args()

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    wanted: list[str] = []
    for variant in (["tiny", "full"] if args.model == "all" else [args.model]):
        wanted += [a for a in MODEL_ASSETS[variant] if a not in wanted]

    failures = []
    for name in wanted:
        try:
            fetch_asset(name, force=args.force)
        except RuntimeError as exc:
            print(f"[fail] {exc}")
            failures.append(name)

    if not args.skip_posture:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        init_posture_checkpoint(force=args.force)

    if not args.skip_test_image:
        fetch_test_image(force=args.force)

    if failures:
        print(f"\nFinished with failures: {', '.join(failures)}")
        if "yolov3.weights" in failures and "yolov3-tiny.weights" not in failures:
            print("YOLOv3-tiny is available; run the pipeline with --model yolov3-tiny.")
        return 1
    print(f"\nAll assets ready in {WEIGHTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
