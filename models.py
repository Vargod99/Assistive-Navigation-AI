"""Model wrappers: YOLOv3 object detector + MobileNetV2 posture classifier.

YOLOv3
------
OpenCV >= 5.0 removed the Darknet importer (``cv2.dnn.readNetFromDarknet``),
so the default backend here is a small native PyTorch implementation that
parses the original Darknet ``.cfg`` and loads the original ``.weights`` file.
It supports every layer type used by ``yolov3.cfg`` and ``yolov3-tiny.cfg``
(convolutional, shortcut, route, upsample, maxpool, yolo) and runs on Apple
Silicon (``mps``), CUDA or CPU. On OpenCV 4.x the ``opencv`` backend can be
selected instead.

Posture
-------
MobileNetV2 (torchvision) with a new ``Dropout -> Linear(1280, 3)`` head that
classifies cropped "person" boxes as standing / sitting / walking.
"""
from __future__ import annotations

import logging
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

POSTURE_CLASSES: tuple[str, ...] = ("sitting", "standing", "walking")  # ImageFolder (alphabetical) order
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PAD_FILL_RGB = tuple(int(round(m * 255)) for m in IMAGENET_MEAN)  # (124, 116, 104)

# pjreddie's coco.names uses VOC-style spellings; map them to readable names
# (these also match the keys of geometry.KNOWN_HEIGHTS_M).
LABEL_ALIASES = {
    "tvmonitor": "tv", "sofa": "couch", "pottedplant": "potted plant",
    "diningtable": "dining table", "motorbike": "motorcycle", "aeroplane": "airplane",
}


def select_device(preference: str = "auto") -> torch.device:
    """``auto`` -> mps (Apple Silicon) -> cuda -> cpu."""
    pref = preference.lower()
    if pref == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if pref == "mps" and not torch.backends.mps.is_available():
        log.warning("MPS requested but unavailable - falling back to CPU")
        return torch.device("cpu")
    if pref == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but unavailable - falling back to CPU")
        return torch.device("cpu")
    return torch.device(pref)


def mps_fully_supported() -> bool:
    """True if MPS can run every op used here (Linear, conv backward, ...).

    Recent PyTorch MPS kernels call ``-[MPSNDArray initWithBuffer:offset:descriptor:]``,
    which only exists on macOS 15+. On macOS 14 those ops abort the process
    (an uncatchable NSException), so callers avoid them there.
    """
    if not torch.backends.mps.is_available():
        return False
    try:
        major = int(platform.mac_ver()[0].split(".")[0])
    except (ValueError, IndexError):
        return False
    return major >= 15


class MatmulLinear(nn.Module):
    """Drop-in inference replacement for ``nn.Linear`` using a plain matmul with a
    pre-transposed contiguous weight (safe on MPS / macOS 14)."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.register_buffer("weight_t", linear.weight.detach().t().contiguous())
        self.register_buffer("bias", linear.bias.detach().clone() if linear.bias is not None else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.contiguous() @ self.weight_t
        return y + self.bias if self.bias is not None else y


def sync(device: torch.device) -> None:
    """Block until queued GPU work is done (for honest latency measurement)."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@dataclass
class Detection:
    box: tuple[int, int, int, int]  # x, y, w, h in original-image pixels
    label: str
    class_id: int
    confidence: float


def parse_darknet_cfg(path: str | Path) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("["):
                blocks.append({"type": line[1:-1].strip()})
            else:
                key, val = line.split("=", 1)
                blocks[-1][key.strip()] = val.strip()
    if not blocks or blocks[0]["type"] not in ("net", "network"):
        raise ValueError(f"{path}: first block must be [net]")
    return blocks


class _MaxPoolSame(nn.Module):
    """Darknet maxpool with stride 1 keeps spatial size (pads right/bottom)."""

    def __init__(self, size: int, stride: int):
        super().__init__()
        self.size, self.stride = size, stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            pad = self.size - 1
            x = F.pad(x, (pad // 2, pad - pad // 2, pad // 2, pad - pad // 2), mode="replicate")
            return F.max_pool2d(x, self.size, 1)
        return F.max_pool2d(x, self.size, self.stride, padding=(self.size - 1) // 2)


class _YoloHead(nn.Module):
    """Decodes raw head output to (B, N, 5 + C) boxes in network-input pixels."""

    def __init__(self, anchors: list[tuple[float, float]], num_classes: int):
        super().__init__()
        self.register_buffer("anchors", torch.tensor(anchors, dtype=torch.float32), persistent=False)
        self.na = len(anchors)
        self.nc = num_classes
        self._grid_cache: dict[tuple, torch.Tensor] = {}

    def _grid(self, h: int, w: int, device, dtype) -> torch.Tensor:
        key = (h, w, device, dtype)
        if key not in self._grid_cache:
            ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=dtype),
                                    torch.arange(w, device=device, dtype=dtype), indexing="ij")
            self._grid_cache[key] = torch.stack((xs, ys), dim=-1).view(1, 1, h, w, 2)
        return self._grid_cache[key]

    def forward(self, x: torch.Tensor, img_size: tuple[int, int]) -> torch.Tensor:
        b, _, h, w = x.shape
        stride = img_size[0] / h
        p = x.view(b, self.na, self.nc + 5, h, w).permute(0, 1, 3, 4, 2)
        grid = self._grid(h, w, x.device, x.dtype)
        xy = (p[..., :2].sigmoid() + grid) * stride
        wh = p[..., 2:4].exp() * self.anchors.to(x.dtype).view(1, self.na, 1, 1, 2)
        rest = p[..., 4:].sigmoid()
        return torch.cat((xy, wh, rest), dim=-1).view(b, -1, self.nc + 5)


class Darknet(nn.Module):
    """Generic Darknet network built from a YOLOv3 / YOLOv3-tiny cfg file."""

    def __init__(self, cfg_path: str | Path):
        super().__init__()
        blocks = parse_darknet_cfg(cfg_path)
        net = blocks[0]
        self.input_size = (int(net.get("height", 416)), int(net.get("width", 416)))
        self.blocks = blocks[1:]
        self.module_list = nn.ModuleList()
        out_channels: list[int] = []
        prev_c = int(net.get("channels", 3))

        for i, blk in enumerate(self.blocks):
            t = blk["type"]
            mod = nn.Sequential()
            if t == "convolutional":
                bn = int(blk.get("batch_normalize", 0))
                filters, k = int(blk["filters"]), int(blk["size"])
                stride = int(blk.get("stride", 1))
                pad = (k - 1) // 2 if int(blk.get("pad", 0)) else 0
                mod.add_module("conv", nn.Conv2d(prev_c, filters, k, stride, pad, bias=not bn))
                if bn:
                    mod.add_module("bn", nn.BatchNorm2d(filters, eps=1e-5, momentum=0.01))
                act = blk.get("activation", "linear")
                if act == "leaky":
                    mod.add_module("act", nn.LeakyReLU(0.1, inplace=True))
                elif act == "relu":
                    mod.add_module("act", nn.ReLU(inplace=True))
                elif act != "linear":
                    raise NotImplementedError(f"activation {act!r}")
                prev_c = filters
            elif t == "maxpool":
                mod.add_module("pool", _MaxPoolSame(int(blk["size"]), int(blk["stride"])))
            elif t == "upsample":
                mod.add_module("up", nn.Upsample(scale_factor=int(blk["stride"]), mode="nearest"))
            elif t == "route":
                layers = [int(v) for v in blk["layers"].split(",")]
                layers = [l if l >= 0 else i + l for l in layers]
                blk["_layers"] = layers  # type: ignore[assignment]
                prev_c = sum(out_channels[l] for l in layers)
            elif t == "shortcut":
                frm = int(blk["from"])
                blk["_from"] = frm if frm >= 0 else i + frm  # type: ignore[assignment]
                if blk.get("activation", "linear") != "linear":
                    raise NotImplementedError("shortcut activation")
            elif t == "yolo":
                mask = [int(v) for v in blk["mask"].split(",")]
                a = [float(v) for v in blk["anchors"].split(",")]
                pairs = [(a[j], a[j + 1]) for j in range(0, len(a), 2)]
                mod = _YoloHead([pairs[m] for m in mask], int(blk["classes"]))  # type: ignore[assignment]
            else:
                raise NotImplementedError(f"Darknet layer type {t!r}")
            self.module_list.append(mod)
            out_channels.append(prev_c)

        # Indices whose outputs are needed later (avoid holding every tensor).
        keep = set()
        for blk in self.blocks:
            keep.update(blk.get("_layers", []))  # type: ignore[arg-type]
            if "_from" in blk:
                keep.add(blk["_from"])  # type: ignore[arg-type]
        self._keep = keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img_size = (x.shape[2], x.shape[3])
        cache: dict[int, torch.Tensor] = {}
        outputs = []
        for i, (blk, mod) in enumerate(zip(self.blocks, self.module_list)):
            t = blk["type"]
            if t in ("convolutional", "maxpool", "upsample"):
                x = mod(x)
            elif t == "route":
                ls = blk["_layers"]
                x = cache[ls[0]] if len(ls) == 1 else torch.cat([cache[l] for l in ls], 1)
            elif t == "shortcut":
                x = x + cache[blk["_from"]]
            elif t == "yolo":
                outputs.append(mod(x, img_size))
            if i in self._keep:
                cache[i] = x
        return torch.cat(outputs, 1)

    def load_darknet_weights(self, path: str | Path) -> None:
        with open(path, "rb") as fh:
            major, minor, _rev = np.fromfile(fh, dtype=np.int32, count=3)
            seen_dtype = np.int64 if (major * 10 + minor) >= 2 else np.int32
            np.fromfile(fh, dtype=seen_dtype, count=1)
            weights = np.fromfile(fh, dtype=np.float32)
        ptr = 0

        def take(t: torch.Tensor) -> None:
            nonlocal ptr
            n = t.numel()
            if ptr + n > weights.size:
                raise ValueError(f"{path}: weights file too short for cfg")
            t.data.copy_(torch.from_numpy(weights[ptr:ptr + n]).view_as(t))
            ptr += n

        for blk, mod in zip(self.blocks, self.module_list):
            if blk["type"] != "convolutional":
                continue
            conv = mod.conv
            if "bn" in mod._modules:
                bn = mod.bn
                for t in (bn.bias, bn.weight, bn.running_mean, bn.running_var):
                    take(t)
            else:
                take(conv.bias)
            take(conv.weight)
        if ptr != weights.size:
            raise ValueError(f"{path}: {weights.size - ptr} unused weights - cfg/weights mismatch")


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio, pad to ``size x size`` with grey (Darknet style)."""
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = cv2.copyMakeBorder(resized, top, size - nh - top, left, size - nw - left,
                             cv2.BORDER_CONSTANT, value=(128, 128, 128))
    return out, scale, (left, top)


class YOLOv3Detector:
    """YOLOv3 / YOLOv3-tiny detector.

    Args:
        cfg, weights, names: Darknet asset paths.
        input_size: network resolution (multiple of 32). 416 = Darknet default;
            320 is faster with modest recall loss for near-range obstacles.
        conf_threshold: min ``objectness * class_prob``.
        nms_threshold: IoU threshold for class-aware NMS.
        backend: ``"torch"`` (default, MPS/CUDA/CPU) or ``"opencv"`` (OpenCV 4.x only).
        classes: optional subset of class names to keep.
    """

    def __init__(
        self,
        cfg: str | Path,
        weights: str | Path,
        names: str | Path,
        input_size: int = 416,
        conf_threshold: float = 0.4,
        nms_threshold: float = 0.45,
        device: str | torch.device = "auto",
        backend: str = "torch",
        half: bool = False,
        classes: Sequence[str] | None = None,
    ):
        if input_size % 32:
            raise ValueError("input_size must be a multiple of 32")
        for p in (cfg, weights, names):
            if not Path(p).is_file():
                raise FileNotFoundError(f"{p} not found - run `python setup_weights.py`")
        raw_names = [n.strip() for n in Path(names).read_text().splitlines() if n.strip()]
        self.names = [LABEL_ALIASES.get(n, n) for n in raw_names]
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        if classes is not None:
            unknown = [c for c in classes if LABEL_ALIASES.get(c, c) not in self.names]
            if unknown:
                raise ValueError(f"unknown class names: {unknown}")
            self.keep_ids = {self.names.index(LABEL_ALIASES.get(c, c)) for c in classes}
        else:
            self.keep_ids = None
        self.backend = backend
        self.device = device if isinstance(device, torch.device) else select_device(device)

        if backend == "torch":
            self.net = Darknet(cfg)
            self.net.load_darknet_weights(weights)
            self.net.eval().to(self.device)
            self.half = half and self.device.type in ("mps", "cuda")
            if self.half:
                self.net.half()
        elif backend == "opencv":
            if not hasattr(cv2.dnn, "readNetFromDarknet"):
                raise RuntimeError(f"OpenCV {cv2.__version__} has no Darknet importer; use backend='torch'")
            self.net = cv2.dnn.readNetFromDarknet(str(cfg), str(weights))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self._out_names = self.net.getUnconnectedOutLayersNames()
            self.half = False
        else:
            raise ValueError(f"unknown backend {backend!r}")

    # Stages are separate methods so the pipeline can time each one.
    def preprocess(self, frame_bgr: np.ndarray):
        boxed, scale, pad = letterbox(frame_bgr, self.input_size)
        meta = (scale, pad, frame_bgr.shape[:2])
        if self.backend == "opencv":
            blob = cv2.dnn.blobFromImage(boxed, 1 / 255.0, swapRB=True, crop=False)
            return blob, meta
        rgb = np.ascontiguousarray(boxed[:, :, ::-1].transpose(2, 0, 1))
        t = torch.from_numpy(rgb).to(self.device, non_blocking=True).unsqueeze(0)
        t = t.half() if self.half else t.float()
        return t.div_(255.0), meta

    @torch.no_grad()
    def infer(self, blob) -> np.ndarray:
        """Forward pass + candidate filtering. Returns (K, 6) [cx,cy,w,h,score,cls] on CPU."""
        if self.backend == "opencv":
            self.net.setInput(blob)
            raw = np.concatenate([o.reshape(-1, o.shape[-1]) for o in self.net.forward(self._out_names)])
            cls_scores = raw[:, 5:]
            cls = cls_scores.argmax(1)
            score = cls_scores[np.arange(len(raw)), cls]  # OpenCV already multiplies by objectness
            m = score >= self.conf_threshold
            return np.column_stack((raw[m, :4] * self.input_size, score[m], cls[m])).astype(np.float32)
        pred = self.net(blob)[0]  # (N, 5 + C)
        obj = pred[:, 4]
        m = obj >= self.conf_threshold  # cheap pre-filter on device
        pred = pred[m]
        cls_prob, cls = pred[:, 5:].max(1)
        score = pred[:, 4] * cls_prob
        out = torch.cat((pred[:, :4], score[:, None], cls[:, None].to(pred.dtype)), 1)
        return out.float().cpu().numpy()

    def postprocess(self, cand: np.ndarray, meta) -> list[Detection]:
        scale, (pad_x, pad_y), (h0, w0) = meta
        cand = cand[cand[:, 4] >= self.conf_threshold]
        if self.keep_ids is not None and len(cand):
            cand = cand[np.isin(cand[:, 5].astype(int), list(self.keep_ids))]
        if not len(cand):
            return []
        cx, cy, bw, bh = (cand[:, i] for i in range(4))
        x1 = np.clip((cx - bw / 2 - pad_x) / scale, 0, w0 - 1)
        y1 = np.clip((cy - bh / 2 - pad_y) / scale, 0, h0 - 1)
        x2 = np.clip((cx + bw / 2 - pad_x) / scale, 0, w0 - 1)
        y2 = np.clip((cy + bh / 2 - pad_y) / scale, 0, h0 - 1)
        boxes = np.column_stack((x1, y1, x2 - x1, y2 - y1))
        scores, cls = cand[:, 4], cand[:, 5].astype(int)
        # Class-aware NMS by offsetting boxes per class.
        offs = cls[:, None] * (max(w0, h0) + 1.0)
        shifted = boxes.copy()
        shifted[:, :2] += offs
        idx = cv2.dnn.NMSBoxes(shifted.tolist(), scores.tolist(), self.conf_threshold, self.nms_threshold)
        idx = np.array(idx).reshape(-1)
        dets = []
        for i in idx[np.argsort(-scores[idx])]:
            x, y, w, h = boxes[i]
            if w < 2 or h < 2:
                continue
            dets.append(Detection((int(round(x)), int(round(y)), int(round(w)), int(round(h))),
                                  self.names[cls[i]], int(cls[i]), float(scores[i])))
        return dets

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        blob, meta = self.preprocess(frame_bgr)
        return self.postprocess(self.infer(blob), meta)


def build_posture_model(num_classes: int = len(POSTURE_CLASSES), pretrained: bool = True,
                        dropout: float = 0.3) -> nn.Module:
    """MobileNetV2 backbone with a fresh classification head."""
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

    weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
    model = mobilenet_v2(weights=weights)
    in_f = model.classifier[-1].in_features
    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
    nn.init.normal_(model.classifier[1].weight, 0, 0.01)
    nn.init.zeros_(model.classifier[1].bias)
    return model


def pad_to_square(img: np.ndarray, fill=PAD_FILL_RGB) -> np.ndarray:
    """Pad an HxWx3 array to square (centred) - preserves body proportions."""
    h, w = img.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    return cv2.copyMakeBorder(img, top, s - h - top, left, s - w - left,
                              cv2.BORDER_CONSTANT, value=fill)


def expand_box(box: Sequence[int], img_w: int, img_h: int, margin: float = 0.08) -> tuple[int, int, int, int]:
    """Grow ``[x, y, w, h]`` by ``margin`` on each side, clipped -> (x1, y1, x2, y2)."""
    x, y, w, h = box
    dx, dy = w * margin, h * margin
    x1, y1 = max(0, int(x - dx)), max(0, int(y - dy))
    x2, y2 = min(img_w, int(x + w + dx)), min(img_h, int(y + h + dy))
    return x1, y1, x2, y2


class PostureClassifier:
    """Batched MobileNetV2 inference on person crops."""

    def __init__(self, checkpoint: str | Path, device: str | torch.device = "auto",
                 img_size: int | None = None):
        self.device = device if isinstance(device, torch.device) else select_device(device)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.classes: list[str] = list(ckpt.get("classes", POSTURE_CLASSES))
        self.trained: bool = bool(ckpt.get("trained", False))
        self.val_acc = ckpt.get("val_acc")
        self.img_size = int(img_size or ckpt.get("img_size", 224))
        self.model = build_posture_model(len(self.classes), pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        if self.device.type == "mps" and not mps_fully_supported():
            self.model.classifier[1] = MatmulLinear(self.model.classifier[1])
        self.model.eval().to(self.device)
        self._mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    def _prepare(self, frame_bgr: np.ndarray, boxes: Sequence[Sequence[int]]) -> torch.Tensor:
        h0, w0 = frame_bgr.shape[:2]
        crops = []
        for b in boxes:
            x1, y1, x2, y2 = expand_box(b, w0, h0)
            crop = frame_bgr[y1:y2, x1:x2, ::-1]  # -> RGB
            if crop.size == 0:
                crop = np.zeros((self.img_size, self.img_size, 3), np.uint8)
            sq = pad_to_square(np.ascontiguousarray(crop))
            crops.append(cv2.resize(sq, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA))
        # Build NCHW contiguously on the host: a permuted (strided) tensor makes
        # PyTorch's MPS conv call an MPSNDArray API that only exists on macOS 15+,
        # which aborts the process on macOS 14.
        nchw = np.ascontiguousarray(np.stack(crops).transpose(0, 3, 1, 2))
        batch = torch.from_numpy(nchw).to(self.device).float().div_(255.0)
        return ((batch - self._mean) / self._std).contiguous()

    @torch.no_grad()
    def predict(self, frame_bgr: np.ndarray, boxes: Sequence[Sequence[int]]) -> list[tuple[str, float]]:
        """Return ``[(label, prob), ...]`` aligned with ``boxes``."""
        if not len(boxes):
            return []
        probs = self.model(self._prepare(frame_bgr, boxes)).softmax(1)
        conf, idx = probs.max(1)
        return [(self.classes[i], float(c)) for i, c in zip(idx.cpu().tolist(), conf.cpu().tolist())]
