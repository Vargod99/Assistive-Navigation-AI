#!/usr/bin/env python3
"""Build a sitting / standing / walking person-crop dataset from public data.



"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
MPII_SAMPLES_URL = "https://huggingface.co/datasets/Voxel51/MPII_Human_Pose_Dataset/resolve/main/samples.json"
MPII_IMG_URL = "https://huggingface.co/datasets/Voxel51/MPII_Human_Pose_Dataset/resolve/main/{}"
CLASSES = ("sitting", "standing", "walking")

COCO_JOINTS = {"l_sh": 5, "r_sh": 6, "l_hip": 11, "r_hip": 12, "l_knee": 13, "r_knee": 14,
               "l_ank": 15, "r_ank": 16}
MPII_JOINTS = {"l_sh": "l_shoulder", "r_sh": "r_shoulder", "l_hip": "l_hip", "r_hip": "r_hip",
               "l_knee": "l_knee", "r_knee": "r_knee", "l_ank": "l_ankle", "r_ank": "r_ankle"}


@dataclass
class Candidate:
    source: str                     # "coco" | "mpii"
    image_key: str                  # unique image id
    url: str
    joints: dict                    # name -> (x, y) pixels (visible joints only)  [COCO]
    joints_norm: dict = field(default_factory=dict)  # name -> (x, y) normalised   [MPII]
    gt_box: tuple | None = None     # COCO person bbox x, y, w, h
    others: list = field(default_factory=list)  # other people's joint lists (for box matching)
    activity: str = ""              # MPII activity or joined COCO captions
    seat_under_hips: bool = False
    sport_overlap: bool = False
    label: str | None = None
    feats: dict = field(default_factory=dict)


def _angle(a, b, c) -> float:
    """Angle ABC in degrees."""
    v1, v2 = np.subtract(a, b), np.subtract(c, b)
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n == 0:
        return float("nan")
    return math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / n, -1, 1))))


def posture_features(j: dict) -> dict | None:
    need = ("l_sh", "r_sh", "l_hip", "r_hip")
    if not all(k in j for k in need):
        return None
    sh = np.mean([j["l_sh"], j["r_sh"]], 0)
    hip = np.mean([j["l_hip"], j["r_hip"]], 0)
    torso = float(np.linalg.norm(sh - hip))
    if torso < 20:
        return None
    dx, dy = hip - sh
    f = {"torso_px": torso, "tilt": abs(math.degrees(math.atan2(dx, dy))) if dy > 0 else 180.0}
    legs = []
    for s in ("l", "r"):
        hip_s, knee, ank = j.get(f"{s}_hip"), j.get(f"{s}_knee"), j.get(f"{s}_ank")
        if hip_s is None or knee is None:
            continue
        leg = {"thigh_drop": (knee[1] - hip_s[1]) / torso,
               "thigh_len": float(np.linalg.norm(np.subtract(knee, hip_s))) / torso}
        if ank is not None:
            leg["knee_angle"] = _angle(hip_s, knee, ank)
            leg["shin_drop"] = (ank[1] - knee[1]) / torso
        legs.append(leg)
    f["n_legs"] = len(legs)
    f["n_full_legs"] = sum("knee_angle" in l for l in legs)
    if legs:
        f["thigh_drop_mean"] = float(np.mean([l["thigh_drop"] for l in legs]))
        f["thigh_drop_min"] = float(min(l["thigh_drop"] for l in legs))
    ka = [l["knee_angle"] for l in legs if "knee_angle" in l]
    if ka:
        f["knee_min"], f["knee_max"] = float(min(ka)), float(max(ka))
    if "l_ank" in j and "r_ank" in j:
        f["stride"] = abs(j["l_ank"][0] - j["r_ank"][0]) / torso
        f["ank_dy"] = abs(j["l_ank"][1] - j["r_ank"][1]) / torso
    return f


_WALK_RE = re.compile(r"\bwalk(s|ing|ed)?\b|\bstroll(s|ing)?\b|\bcrossing\b|\bwalkway\b")
_SIT_RE = re.compile(r"\bsit(s|ting)?\b|\bseated\b")
_STAND_RE = re.compile(r"\bstand(s|ing)?\b")


def label_from_geometry(f: dict | None, source: str, context: str,
                        seat_under_hips: bool = False, sport_overlap: bool = False) -> str | None:
    """Conservative rules: return None unless the pose is clearly one class.

    ``context`` is the MPII activity name or the joined COCO captions. A single
    frame can't separate walking from sports poses / wide stances, so walking
    always needs the context to mention walking; COCO sitting needs a seat
    object under the hips or a caption saying "sitting".
    """
    if f is None or f["tilt"] > 35 or f["n_legs"] == 0 or sport_overlap:
        return None
    ctx = context.lower()
    says_walk = bool(_WALK_RE.search(ctx))
    says_sit = bool(_SIT_RE.search(ctx))
    says_stand = bool(_STAND_RE.search(ctx))

    # Sitting: thighs near horizontal, knees bent (if ankles visible).
    if f["thigh_drop_mean"] < 0.35 and f.get("knee_min", 90) < 135 and f["tilt"] < 30:
        if source == "mpii":
            return None if (says_walk or (says_stand and not says_sit)) else "sitting"
        return "sitting" if (seat_under_hips or says_sit) else None

    # Upright legs required for standing / walking.
    if f["n_full_legs"] < 2 or f["tilt"] > 20 or f["thigh_drop_min"] < 0.6:
        return None
    stride, ank_dy = f.get("stride", 0.0), f.get("ank_dy", 0.0)

    if f["knee_min"] > 165 and stride < 0.30 and ank_dy < 0.12:
        if says_walk or (source == "mpii" and says_sit and not says_stand):
            return None
        return "standing"

    stepping = (stride > 0.35 or ank_dy > 0.15) and 115 < f["knee_min"] < 175
    if stepping and f["tilt"] < 15 and says_walk:
        return "walking"
    return None


def _download(url: str, dest: Path, timeout: int = 60) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "posture-dataset-builder/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as fh:
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
            tmp.replace(dest)
            return True
        except Exception:
            time.sleep(1 + attempt * 2)
    tmp.unlink(missing_ok=True)
    return False


SEAT_CLASSES = {"chair", "couch", "bench", "bed", "toilet"}
# People overlapping these are usually doing sports / riding: poses don't transfer.
SPORT_CLASSES = {"sports ball", "tennis racket", "baseball bat", "baseball glove", "skis",
                 "snowboard", "skateboard", "surfboard", "frisbee", "kite", "bicycle",
                 "motorcycle", "horse", "elephant"}


def _box_contains(b, p) -> bool:
    return b[0] <= p[0] <= b[0] + b[2] and b[1] <= p[1] <= b[1] + b[3]


def load_coco(raw: Path, min_height: int) -> list[Candidate]:
    zpath = raw / "coco" / "annotations_trainval2017.zip"
    if not zpath.exists():
        print(f"downloading {COCO_ANN_URL} (~250 MB)")
        if not _download(COCO_ANN_URL, zpath, timeout=600):
            raise SystemExit("COCO annotation download failed")
    cands = []
    with zipfile.ZipFile(zpath) as zf:
        for split in ("train2017", "val2017"):
            data = json.loads(zf.read(f"annotations/person_keypoints_{split}.json"))
            caps = defaultdict(list)
            for a in json.loads(zf.read(f"annotations/captions_{split}.json"))["annotations"]:
                caps[a["image_id"]].append(a["caption"])
            inst = json.loads(zf.read(f"annotations/instances_{split}.json"))
            cat = {c["id"]: c["name"] for c in inst["categories"]}
            seats, sports = defaultdict(list), defaultdict(list)
            for a in inst["annotations"]:
                name = cat[a["category_id"]]
                if name in SEAT_CLASSES:
                    seats[a["image_id"]].append(a["bbox"])
                elif name in SPORT_CLASSES:
                    sports[a["image_id"]].append(a["bbox"])
            del inst
            imgs = {im["id"]: im for im in data["images"]}
            by_img = defaultdict(list)
            for a in data["annotations"]:
                if not a.get("iscrowd") and a["num_keypoints"] > 0:
                    by_img[a["image_id"]].append(a)
            for img_id, anns in by_img.items():
                if len(anns) > 6:  # crowded scenes -> ambiguous YOLO matching
                    continue
                kp_lists = []
                for a in anns:
                    k = np.array(a["keypoints"], float).reshape(-1, 3)
                    kp_lists.append({n: (k[i, 0], k[i, 1]) for n, i in COCO_JOINTS.items() if k[i, 2] == 2})
                for a, j in zip(anns, kp_lists):
                    if a["num_keypoints"] < 11 or a["bbox"][3] < min_height:
                        continue
                    im = imgs[img_id]
                    hips = [j[k] for k in ("l_hip", "r_hip") if k in j]
                    hip = tuple(np.mean(hips, 0)) if hips else None
                    seat = hip is not None and any(_box_contains(b, hip) for b in seats[img_id])
                    sport = any(_iou(a["bbox"], b) > 0.05 or _box_contains(a["bbox"], (b[0] + b[2] / 2, b[1] + b[3] / 2))
                                for b in sports[img_id])
                    cands.append(Candidate(
                        "coco", f"coco_{img_id}", im["coco_url"], j, gt_box=tuple(a["bbox"]),
                        others=[o for o in kp_lists if o is not j], activity=" | ".join(caps[img_id]),
                        seat_under_hips=seat, sport_overlap=sport))
    return cands


MPII_POSTURE_WORDS = ("walk", "sitting", "standing", "inactivity", "queue", "stroll")


def load_mpii(raw: Path, all_activities: bool = False) -> list[Candidate]:
    spath = raw / "mpii" / "samples.json"
    if not spath.exists():
        print(f"downloading {MPII_SAMPLES_URL} (~63 MB)")
        if not _download(MPII_SAMPLES_URL, spath, timeout=600):
            raise SystemExit("MPII samples.json download failed")
    samples = json.loads(spath.read_text())["samples"]
    cands = []
    for s in samples:
        labels = [c.get("label", "") for c in s["activity"]["classifications"] if c.get("label")]
        if not labels:
            continue
        activity = " | ".join(labels)
        if not all_activities and not any(w in activity.lower() for w in MPII_POSTURE_WORDS):
            continue
        people = []
        for kp in s["annopoints"]["keypoints"]:
            names, pts, vis = kp.get("joints_id") or [], kp["points"], kp.get("visible") or []
            jn = {}
            for ours, theirs in MPII_JOINTS.items():
                if theirs in names:
                    i = names.index(theirs)
                    # MPII 'visible' is often null for visible joints; only reject explicit False
                    if i < len(pts) and (i >= len(vis) or vis[i] is not False):
                        jn[ours] = tuple(pts[i])
            people.append(jn)
        if not people or len(people) > 6:
            continue
        for jn in people:
            if len(jn) < 6:
                continue
            cands.append(Candidate("mpii", f"mpii_{Path(s['filepath']).stem}",
                                   MPII_IMG_URL.format(s["filepath"]), {}, joints_norm=jn,
                                   others=[o for o in people if o is not jn], activity=activity))
    return cands


def _inside(box, pts) -> float:
    if not pts:
        return 0.0
    x, y, w, h = box
    return sum(x <= p[0] <= x + w and y <= p[1] <= y + h for p in pts) / len(pts)


def _iou(a, b) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def match_box(c: Candidate, det_boxes: list, joints: dict, others: list) -> tuple | None:
    best, best_score = None, 0.0
    pts = list(joints.values())
    for b in det_boxes:
        if c.gt_box is not None:
            score = _iou(b, c.gt_box)
            if score < 0.6:
                continue
        else:
            frac = _inside(b, pts)
            if frac < 0.9:
                continue
            if any(_inside(b, list(o.values())) > 0.5 for o in others if o):
                continue  # box swallows another annotated person
            score = frac
        if score > best_score:
            best, best_score = b, score
    return best


def contact_sheet(paths: list[Path], dest: Path, cols: int = 10, cell: int = 128, n: int = 80) -> None:
    paths = paths[:n]
    if not paths:
        return
    rows = math.ceil(len(paths) / cols)
    sheet = np.full((rows * cell, cols * cell, 3), 255, np.uint8)
    for i, p in enumerate(paths):
        im = cv2.imread(str(p))
        if im is None:
            continue
        s = cell / max(im.shape[:2])
        im = cv2.resize(im, (max(1, int(im.shape[1] * s)), max(1, int(im.shape[0] * s))))
        r, c = divmod(i, cols)
        y0, x0 = r * cell + (cell - im.shape[0]) // 2, c * cell + (cell - im.shape[1]) // 2
        sheet[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
    cv2.imwrite(str(dest), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/posture")
    ap.add_argument("--raw", default="data/raw", help="annotation + image download cache")
    ap.add_argument("--sources", nargs="+", default=["coco", "mpii"], choices=["coco", "mpii"])
    ap.add_argument("--per-class", type=int, default=1500, help="max crops per class (train + val)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--min-height", type=int, default=96, help="min person height in px")
    ap.add_argument("--workers", type=int, default=12, help="parallel image downloads")
    ap.add_argument("--max-side", type=int, default=320, help="resize saved crops (longer side)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mpii-all-activities", action="store_true",
                    help="consider every MPII image, not just posture-related activities")
    ap.add_argument("--dry-run", action="store_true", help="only report candidate counts")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    raw, out = Path(args.raw), Path(args.out)

    cands: list[Candidate] = []
    if "coco" in args.sources:
        coco = load_coco(raw, args.min_height)
        for c in coco:
            c.feats = posture_features(c.joints) or {}
            c.label = label_from_geometry(c.feats or None, "coco", c.activity,
                                          c.seat_under_hips, c.sport_overlap)
        cands += [c for c in coco if c.label]
        print(f"COCO: {len(coco)} persons -> {Counter(c.label for c in coco if c.label)}")
    if "mpii" in args.sources:
        mpii = load_mpii(raw, args.mpii_all_activities)
        # MPII coords are normalised; geometry needs pixels, so label after download.
        cands += mpii
        print(f"MPII: {len(mpii)} annotated persons (labelled after download)")

    # Cap COCO per class before downloading (oversample 1.6x for match failures);
    # all MPII candidates are kept since they're labelled later.
    by_cls = defaultdict(list)
    for c in cands:
        if c.source == "coco":
            by_cls[c.label].append(c)
    keep = [c for c in cands if c.source == "mpii"]
    for cls, lst in by_cls.items():
        rng.shuffle(lst)
        keep += lst[: int(args.per_class * 1.6)]
    if args.dry_run:
        print("dry run - images to download:", len({c.image_key for c in keep}))
        return 0

    urls = {}
    for c in keep:
        urls[c.image_key] = (c.url, raw / c.source / "images" / f"{c.image_key}.jpg")
    todo = [(u, p) for u, p in urls.values() if not p.exists()]
    print(f"downloading {len(todo)} images ({len(urls) - len(todo)} cached) with {args.workers} threads")
    t0, done, failed = time.time(), 0, 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(_download, u, p) for u, p in todo]
        for f in as_completed(futs):
            done += 1
            failed += not f.result()
            if done % 250 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)} ({failed} failed) {time.time() - t0:.0f}s", flush=True)

    sys.path.insert(0, str(ROOT))
    from models import YOLOv3Detector, expand_box

    det = YOLOv3Detector(ROOT / "weights/yolov3.cfg", ROOT / "weights/yolov3.weights",
                         ROOT / "weights/coco.names", classes=["person"], conf_threshold=0.3)
    by_image = defaultdict(list)
    for c in keep:
        by_image[c.image_key].append(c)

    crops_dir = raw / "crops"
    results = []  # (label, image_key, crop_path, cand)
    stats = Counter()
    t0 = time.time()
    for n, (key, cs) in enumerate(by_image.items(), 1):
        img_path = urls[key][1]
        img = cv2.imread(str(img_path)) if img_path.exists() else None
        if img is None:
            stats["unreadable"] += len(cs)
            continue
        h, w = img.shape[:2]
        for c in cs:
            if c.source == "mpii":
                c.joints = {k: (x * w, y * h) for k, (x, y) in c.joints_norm.items()}
                c.others = [{k: (x * w, y * h) for k, (x, y) in o.items()} for o in c.others]
                c.feats = posture_features(c.joints) or {}
                c.label = label_from_geometry(c.feats or None, "mpii", c.activity)
        cs = [c for c in cs if c.label]
        if not cs:
            continue
        boxes = [d.box for d in det.detect(img)]
        for i, c in enumerate(cs):
            b = match_box(c, boxes, c.joints, c.others)
            if b is None:
                stats["no_yolo_match"] += 1
                continue
            if b[3] < args.min_height:
                stats["too_small"] += 1
                continue
            x1, y1, x2, y2 = expand_box(b, w, h)
            crop = img[y1:y2, x1:x2]
            s = args.max_side / max(crop.shape[:2])
            if s < 1:
                crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)),
                                  interpolation=cv2.INTER_AREA)
            cp = crops_dir / c.label / f"{key}_{i}.jpg"
            cp.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(cp), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            results.append((c.label, key, cp, c))
            stats[f"ok_{c.source}_{c.label}"] += 1
        if n % 500 == 0:
            print(f"  processed {n}/{len(by_image)} images, {len(results)} crops, "
                  f"{time.time() - t0:.0f}s", flush=True)
    print("crop stats:", dict(sorted(stats.items())))

    per_cls = defaultdict(list)
    for r in results:
        per_cls[r[0]].append(r)
    if out.exists():
        for cls in CLASSES:
            for split in ("train", "val"):
                for f in (out / split / cls).glob("*.jpg"):
                    f.unlink()
    images = sorted({r[1] for r in results})
    rng.shuffle(images)
    val_imgs = set(images[: int(len(images) * args.val_frac)])
    manifest = []
    n_target = min(args.per_class, min((len(v) for v in per_cls.values()), default=0) * 3)
    for cls in CLASSES:
        lst = per_cls.get(cls, [])
        # prefer MPII-labelled walking/sitting (activity-confirmed) by not shuffling them out
        rng.shuffle(lst)
        lst = lst[: n_target]
        for label, key, cp, c in lst:
            split = "val" if key in val_imgs else "train"
            dest = out / split / cls / cp.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(cp.read_bytes())
            manifest.append({"split": split, "label": cls, "file": str(dest.relative_to(out)),
                             "source": c.source, "image": key, "activity": c.activity,
                             **{k: round(v, 3) for k, v in c.feats.items()}})

    keys = sorted({k for m in manifest for k in m})
    with open(out / "manifest.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(manifest)
    for cls in CLASSES:
        files = sorted((out / "train" / cls).glob("*.jpg"))
        rng.shuffle(files)
        contact_sheet(files, out / f"review_{cls}.jpg")

    cnt = Counter((m["split"], m["label"]) for m in manifest)
    src = Counter((m["label"], m["source"]) for m in manifest)
    print("\nDataset written to", out)
    for cls in CLASSES:
        print(f"  {cls:<9} train {cnt[('train', cls)]:5d}  val {cnt[('val', cls)]:4d}   "
              f"(coco {src[(cls, 'coco')]}, mpii {src[(cls, 'mpii')]})")
    print(f"Review: {out}/review_<class>.jpg ; then python train_posture.py --data-dir {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
