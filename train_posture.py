#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

from models import (IMAGENET_MEAN, IMAGENET_STD, PAD_FILL_RGB, build_posture_model,
                    expand_box, mps_fully_supported, select_device)
from geometry import POSTURE_HEIGHTS_M

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class PadToSquare:
    """PIL equivalent of ``models.pad_to_square`` (keeps body proportions)."""

    def __init__(self, fill=PAD_FILL_RGB):
        self.fill = tuple(fill)

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        s = max(w, h)
        if w == h:
            return img
        canvas = Image.new("RGB", (s, s), self.fill)
        canvas.paste(img, ((s - w) // 2, (s - h) // 2))
        return canvas


def build_transforms(img_size: int):
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train_tf = transforms.Compose([
        PadToSquare(),
        transforms.RandomResizedCrop(img_size, scale=(0.70, 1.0), ratio=(0.85, 1.15), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=8, translate=(0.05, 0.05), fill=PAD_FILL_RGB),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        norm,
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
    ])
    val_tf = transforms.Compose([
        PadToSquare(),
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.ToTensor(),
        norm,
    ])
    return train_tf, val_tf


def _rgb_loader(path: str) -> Image.Image:
    with open(path, "rb") as fh:
        return Image.open(fh).convert("RGB")


def _stratified_split(targets: list[int], val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    by_cls: dict[int, list[int]] = {}
    for i, t in enumerate(targets):
        by_cls.setdefault(t, []).append(i)
    tr, va = [], []
    for idx in by_cls.values():
        rng.shuffle(idx)
        k = max(1, int(round(len(idx) * val_frac))) if len(idx) > 1 else 0
        va += idx[:k]
        tr += idx[k:]
    return tr, va


def build_datasets(root: Path, img_size: int, val_split: float, seed: int):
    train_tf, val_tf = build_transforms(img_size)
    train_dir = root / "train" if (root / "train").is_dir() else root
    val_dir = root / "val"
    if val_dir.is_dir():
        tr = datasets.ImageFolder(train_dir, train_tf, loader=_rgb_loader)
        va = datasets.ImageFolder(val_dir, val_tf, loader=_rgb_loader)
        if va.classes != tr.classes:
            raise SystemExit(f"class mismatch: train={tr.classes} val={va.classes}")
        return tr, va, tr.classes, list(tr.targets)
    base_tr = datasets.ImageFolder(train_dir, train_tf, loader=_rgb_loader)
    base_va = datasets.ImageFolder(train_dir, val_tf, loader=_rgb_loader)
    tr_idx, va_idx = _stratified_split(base_tr.targets, val_split, seed)
    print(f"No val/ folder - stratified {val_split:.0%} split of {train_dir}")
    return (Subset(base_tr, tr_idx), Subset(base_va, va_idx), base_tr.classes,
            [base_tr.targets[i] for i in tr_idx])


def crop_persons(src: Path, dst: Path, weights_dir: Path, min_height: int, all_persons: bool) -> None:
    import cv2

    from models import YOLOv3Detector

    det = YOLOv3Detector(weights_dir / "yolov3.cfg", weights_dir / "yolov3.weights",
                         weights_dir / "coco.names", classes=["person"], conf_threshold=0.5)
    classes = sorted(p.name for p in src.iterdir() if p.is_dir())
    if not classes:
        raise SystemExit(f"{src} must contain one sub-folder per posture class")
    for cls in classes:
        out = dst / cls
        out.mkdir(parents=True, exist_ok=True)
        n_in = n_out = 0
        for f in sorted((src / cls).rglob("*")):
            if f.suffix.lower() not in IMG_EXTS:
                continue
            img = cv2.imread(str(f))
            if img is None:
                continue
            n_in += 1
            dets = sorted((d for d in det.detect(img) if d.box[3] >= min_height),
                          key=lambda d: -d.box[2] * d.box[3])
            for k, d in enumerate(dets if all_persons else dets[:1]):
                x1, y1, x2, y2 = expand_box(d.box, img.shape[1], img.shape[0])
                cv2.imwrite(str(out / f"{f.stem}_p{k}.jpg"), img[y1:y2, x1:x2])
                n_out += 1
        print(f"  {cls:<10} {n_in:5d} images -> {n_out:5d} crops")


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for p in model.features.parameters():
        p.requires_grad = trainable


def freeze_bn(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def warmup_cosine(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int):
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    loss_sum, n = 0.0, 0
    crit = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += crit(logits, y).item() * len(y)
        n += len(y)
        pred = logits.argmax(1)
        for t, p in zip(y.cpu(), pred.cpu()):
            cm[t, p] += 1
    acc = cm.diag().sum().item() / max(1, cm.sum().item())
    return acc, loss_sum / max(1, n), cm


def train(args) -> float:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = select_device(args.device)
    if device.type == "mps" and not mps_fully_supported():
        print("NOTE: MPS training needs macOS 15+ with this PyTorch build "
              "(conv backward / Linear abort on older macOS) - training on CPU.")
        device = torch.device("cpu")
    print(f"Device: {device}")

    tr_ds, va_ds, classes, tr_targets = build_datasets(Path(args.data_dir), args.img_size,
                                                       args.val_split, args.seed)
    unknown = [c for c in classes if c not in POSTURE_HEIGHTS_M]
    if unknown:
        print(f"WARNING: classes {unknown} have no posture height in geometry.POSTURE_HEIGHTS_M; "
              "the pipeline will use the default person height for them.")
    counts = Counter(tr_targets)
    print(f"Classes: {classes}")
    print(f"Train: {len(tr_ds)} ({', '.join(f'{classes[k]}={v}' for k, v in sorted(counts.items()))})"
          f" | Val: {len(va_ds)}")
    if len(va_ds) == 0:
        raise SystemExit("validation set is empty")

    class_w = {c: 1.0 / n for c, n in counts.items()}
    sampler = WeightedRandomSampler([class_w[t] for t in tr_targets], num_samples=len(tr_targets),
                                    replacement=True, generator=torch.Generator().manual_seed(args.seed))
    pin = device.type == "cuda"
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                       pin_memory=pin, drop_last=len(tr_ds) > args.batch_size,
                       persistent_workers=args.workers > 0)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers,
                       pin_memory=pin, persistent_workers=args.workers > 0)

    model = build_posture_model(len(classes), pretrained=not args.no_pretrained,
                                dropout=args.dropout)
    if args.init_checkpoint:
        init = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if list(init.get("classes", classes)) != list(classes):
            raise SystemExit(f"class mismatch with {args.init_checkpoint}: {init.get('classes')} vs {classes}")
        model.load_state_dict(init["state_dict"])
        print(f"Initialised from {args.init_checkpoint} (val_acc={init.get('val_acc')})")
    model = model.to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best_acc, best_epoch, stale = -1.0, -1, 0
    steps_per_epoch = max(1, len(tr_dl))

    def make_opt(phase: int):
        if phase == 1:
            params = [{"params": model.classifier.parameters(), "lr": args.lr_head}]
            epochs = args.freeze_epochs
        else:
            params = [{"params": model.features.parameters(), "lr": args.lr_backbone},
                      {"params": model.classifier.parameters(), "lr": args.lr_backbone * 10}]
            epochs = args.epochs - args.freeze_epochs
        opt = torch.optim.AdamW(params, weight_decay=args.weight_decay)
        total = max(1, epochs * steps_per_epoch)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: warmup_cosine(s, total, min(steps_per_epoch, total // 10)))
        return opt, sched

    phase = 1 if args.freeze_epochs > 0 else 2
    set_backbone_trainable(model, phase == 2)
    opt, sched = make_opt(phase)

    for epoch in range(args.epochs):
        if phase == 1 and epoch >= args.freeze_epochs:
            phase = 2
            set_backbone_trainable(model, True)
            opt, sched = make_opt(2)
            print("-- unfreezing backbone (phase 2) --")
        model.train()
        if phase == 1:
            freeze_bn(model.features)
        t0, run_loss, correct, seen = time.time(), 0.0, 0, 0
        for x, y in tr_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
        va_acc, va_loss, cm = evaluate(model, va_dl, device, len(classes))
        print(f"epoch {epoch + 1:3d}/{args.epochs} [p{phase}] train loss {run_loss / seen:.4f} "
              f"acc {correct / seen:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f} | {time.time() - t0:.1f}s")
        if va_acc > best_acc:
            best_acc, best_epoch, stale = va_acc, epoch + 1, 0
            torch.save({
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "classes": classes,
                "img_size": args.img_size,
                "arch": "mobilenet_v2",
                "trained": True,
                "val_acc": va_acc,
                "epoch": epoch + 1,
                "confusion_matrix": cm.tolist(),
                "args": vars(args),
            }, out)
        else:
            stale += 1
            if phase == 2 and stale >= args.patience:
                print(f"early stop: no val improvement for {args.patience} epochs")
                break

    ckpt = torch.load(out, map_location="cpu", weights_only=False)
    cm = torch.tensor(ckpt["confusion_matrix"])
    print(f"\nBest val accuracy {best_acc:.3f} at epoch {best_epoch} -> {out}")
    print("Confusion matrix (rows = true, cols = predicted):")
    print("            " + "".join(f"{c[:9]:>10}" for c in classes))
    for i, c in enumerate(classes):
        row_acc = cm[i, i].item() / max(1, cm[i].sum().item())
        print(f"  {c[:9]:<10}" + "".join(f"{v:10d}" for v in cm[i].tolist()) + f"   acc {row_acc:.3f}")
    if best_acc < args.target_acc:
        print(f"\nBelow the {args.target_acc:.0%} target. Typical fixes: more / cleaner data "
              "(>=300 crops per class), crops via --crop-persons, more epochs, --img-size 256.")
    return best_acc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/posture")
    ap.add_argument("--out", default="weights/mobilenetv2_posture.pth")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--freeze-epochs", type=int, default=5, help="head-only epochs before unfreezing")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--target-acc", type=float, default=0.90)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init-checkpoint", help="resume weights from a previous checkpoint "
                    "(use with --freeze-epochs 0 to continue fine-tuning)")
    ap.add_argument("--no-pretrained", action="store_true", help="train from scratch (not recommended)")
    ap.add_argument("--crop-persons", nargs=2, metavar=("SRC", "DST"),
                    help="crop YOLOv3 person boxes from SRC/<class>/* into DST/<class>/ and exit")
    ap.add_argument("--all-persons", action="store_true", help="with --crop-persons: keep every person, not just the largest")
    ap.add_argument("--min-person-height", type=int, default=64)
    ap.add_argument("--weights-dir", default="weights")
    args = ap.parse_args()
    # Show progress immediately even when output is redirected to a log file.
    sys.stdout.reconfigure(line_buffering=True)

    if args.crop_persons:
        crop_persons(Path(args.crop_persons[0]), Path(args.crop_persons[1]), Path(args.weights_dir),
                     args.min_person_height, args.all_persons)
        return 0
    if not Path(args.data_dir).is_dir():
        raise SystemExit(f"dataset folder {args.data_dir!r} not found (see --help for the layout)")
    if args.freeze_epochs >= args.epochs and args.epochs > 0:
        raise SystemExit("--freeze-epochs must be smaller than --epochs")
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
