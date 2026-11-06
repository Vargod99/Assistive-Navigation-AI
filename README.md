# Assistive Navigation Vision System

![tests](https://github.com/Vargod99/BlindUserObjDetect/actions/workflows/tests.yml/badge.svg)

Turns a camera frame into a short spoken movement instruction for blind and
low-vision users, e.g.

> **Caution. Chair at 12 o'clock, 1.5 meters. Veer right.**

```
frame ─► resize/letterbox ─► YOLOv3 (obstacles, people) ─► NMS
      ─► MobileNetV2 posture on person crops (sitting / standing / walking)
      ─► geometry: clock direction + pinhole distance + lateral extent
      ─► planner: walking-corridor check ─► instruction (+ optional speech)
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python setup_weights.py                      # YOLOv3 + YOLOv3-tiny, posture ckpt, test_scene.jpg
python pipeline.py --image test_scene.jpg    # scene, instruction, per-stage latency, PASS/FAIL
```

Other modes:

```bash
python pipeline.py --image test_scene.jpg --runs 30 --save annotated.jpg --json
python pipeline.py --camera 0 --speak --show        # live webcam, macOS `say` / Linux `espeak`
python pipeline.py --camera 1 --show --flip mirror  # mirrored, selfie-style view (also: vertical, rotate180)
python pipeline.py --video walk.mp4 --save out.mp4
python pipeline.py --image x.jpg --model yolov3-tiny --device cpu --strict   # exit 1 if over budget
```

## Files

| File | Purpose |
|---|---|
| `geometry.py` | `CameraModel` (pinhole from HFOV), bearing to clock face (9–3 o'clock), distance `Z = H·f/h`, lateral extent, speech formatting |
| `models.py` | Native PyTorch Darknet loader (`yolov3.cfg` + `.weights`), `YOLOv3Detector`, `PostureClassifier` (MobileNetV2) |
| `pipeline.py` | `NavigationPipeline` with per-stage `time.perf_counter()` timing, `NavigationPlanner`, `ApproachTracker`, `InstructionSmoother`, CLI |
| `train_posture.py` | Two-phase MobileNetV2 fine-tuning, plus `--crop-persons` to build crops with YOLOv3 |
| `build_posture_dataset.py` | Builds a sitting/standing/walking crop dataset from COCO 2017 and MPII |
| `tests/` | Unit tests for geometry, the planner, the smoother, the Darknet loader and the labelling rules. They need no weights. |
| `setup_weights.py` | Downloads YOLO assets with mirrors and size checks, initialises `mobilenetv2_posture.pth`, fetches `test_scene.jpg` |

## Design notes

**YOLOv3 backend.** OpenCV 5 removed `cv2.dnn.readNetFromDarknet`, so
`models.Darknet` parses the original cfg and loads the original weights
directly in PyTorch. That code path runs on `mps`, `cuda` or `cpu` under
`torch.no_grad()`. On OpenCV 4.x, `YOLOv3Detector(..., backend="opencv")` is
also available. Inputs are letterboxed to `--input-size` (default 416; 320 is
faster).

**Clock directions.** `--clock-mode angular` (the default) is physically
correct: hour = 12 + round(bearing / 30°). With a 60° camera the edges of the
frame are ±30°, so you only get 11, 12 and 1 o'clock. If the user turns toward
the announced hour, they face the object. `--clock-mode fov` instead spreads
the whole field of view linearly over 9–3 o'clock. That gives finer relative
layout, but the hour is no longer a true heading. Both are clamped to 9–3.

**Distance.** `Z = H_real · f_px / h_px`, where `f_px = (W/2)/tan(HFOV/2)`.
Real-world heights come from `geometry.KNOWN_HEIGHTS_M`. For people, height
depends on posture (standing or walking 1.70 m, sitting 1.30 m), but only
once a trained posture model is loaded. Boxes cut off by the top or bottom of
the frame are reported as upper bounds ("within 2 meters"). **Set `--hfov`
to your camera's real FOV.** Every distance scales with it.

**Planner.** A 0.9 m wide corridor straight ahead is checked out to 3 m. The
nearest object in that corridor becomes the subject: under 1 m gives "Stop.",
under 2 m gives "Caution.". The planner then tests corridors just left and
right of that object and advises "Veer/Step left/right". If both are blocked,
it says "Path blocked". By default only obstacle-relevant classes are
considered (`--announce-all` includes everything).

**Video.** `ApproachTracker` matches people across frames by IoU and fits the
slope of distance over time, flagging "approaching" when the person closes
faster than 0.35 m/s. `InstructionSmoother` requires a new action to persist
for 3 frames before announcing it, except an escalation to "Stop", which is
announced immediately.

## Posture model

`setup_weights.py` creates `weights/mobilenetv2_posture.pth` with an
**ImageNet backbone and an untrained 3-class head** (`trained=False`). The
pipeline still runs it, so the latency numbers are real, but it **does not
announce posture labels**: they'd be random until you fine-tune
it or download the trained weights from Releases (see below).

**Building a dataset from public data.** `build_posture_dataset.py` assembles
a crop dataset from **COCO 2017 person keypoints** and **MPII Human Pose**
(using the Voxel51 Hugging Face mirror). It downloads only the images it needs.

- **Labels:** each person is labelled from joint geometry (knee angles, thigh
  orientation, stride, torso tilt). MPII activity names veto contradictions,
  and MPII walking must come from a walk-related activity.
- **Crops:** made with the pipeline's own YOLOv3 box, so they match what the
  model sees at inference time.
- **Split:** train/val is split by source image.
- **Review:** it writes `manifest.csv` and `review_<class>.jpg` contact sheets.
  The labels are heuristic, so spot-check them before training.
- **License:** both sources are research / non-commercial.

```bash
python build_posture_dataset.py            # ~7.5k images downloaded to data/raw/
python train_posture.py --data-dir data/posture
```

The expected folder layout, if you bring your own data:

```
data/posture/train/{sitting,standing,walking}/*.jpg
data/posture/val/{sitting,standing,walking}/*.jpg     # optional; else --val-split 0.2

python train_posture.py --crop-persons raw_photos/ data/posture/train   # optional: YOLO person crops
python train_posture.py --data-dir data/posture --epochs 30
```

The training recipe:
- pad-to-square, identical to inference
- strong augmentation
- class-balanced sampling
- label smoothing of 0.1
- 5 head-only epochs with BatchNorm frozen, then full fine-tuning with
  discriminative learning rates, AdamW and warm-up plus cosine decay
- early stopping on best validation accuracy, then a confusion matrix

This recipe is built to reach about 90% validation accuracy, given clean,
balanced data of roughly 300 or more crops per class. That figure depends on
your dataset, and none ships with this repo. "Walking" from a single frame is
inherently the hardest class. In video, the tracker's "approaching" signal
complements it.

### Posture results

Trained on 2,997 crops (1,269 sitting, 802 standing, 926 walking) built with
`build_posture_dataset.py` from COCO 2017 and MPII, and validated on 529
held-out crops split by image. The run used 4 head-only epochs, then 11 epochs
of full fine-tuning on CPU.

**Best validation accuracy: 90.4%** (val loss 0.283).

| true \ predicted | sitting | standing | walking | recall |
|---|---|---|---|---|
| sitting | 223 | 3 | 5 | 96.5% |
| standing | 2 | 111 | 20 | 83.5% |
| walking | 1 | 20 | 144 | 87.3% |

Nearly all of the remaining errors are standing vs. walking, which is
ambiguous from a single frame. Sitting is reliable. The validation labels come
from the same geometric heuristics as the training labels, so treat 90% as
agreement with those rules, not with human annotation.

The trained weights (`mobilenetv2_posture.pth`) are attached to the GitHub
Release for **research and non-commercial use only**. Put the file in
`weights/` to replace the untrained checkpoint.

## Tests

```bash
pip install pytest && python -m pytest -q tests
```

GitHub Actions runs the same tests on CPU for every push.

## Measured latency (Apple M3 Pro, macOS 14.1, YOLOv3 @ 416, test_scene.jpg)

| Config | p50 total | p95 total | Budget |
|---|---|---|---|
| YOLOv3, `mps` | ~35–39 ms | ~35–41 ms | 300 ms ✅ |
| YOLOv3, `cpu` | ~151 ms | ~193 ms | 300 ms ✅ |
| YOLOv3-tiny, `mps` | ~17 ms | ~18 ms | 300 ms ✅ |

The figures are steady-state. `NavigationPipeline` warms up at start-up for
every posture batch size, because MPS compiles one graph per shape. The very
first warm-up frame takes about 0.5–0.75 s and is reported separately as
"cold first frame". Model loading takes about 2 s.

## Platform notes

* **macOS 14 + recent PyTorch on MPS:** some MPS kernels (`nn.Linear`,
  strided conv inputs, conv backward) call an API that only exists on macOS
  15+, and the process aborts with
  `-[MPSNDArray initWithBuffer:offset:descriptor:]: unrecognized selector`.
  The project works around this:
  * inference feeds contiguous tensors and swaps the posture head for a
    matmul (`models.MatmulLinear`)
  * `train_posture.py` trains on CPU when MPS is on macOS < 15
* `--show` needs the GUI OpenCV build (`opencv-python`, not `-headless`).

## Credits and licenses

- **Code:** MIT (see `LICENSE`).
- **YOLOv3:** Redmon & Farhadi, 2018. Configs and weights come from
  [pjreddie/darknet](https://github.com/pjreddie/darknet) and are downloaded
  at setup.
- **MobileNetV2:** Sandler et al., 2018. ImageNet weights come from torchvision.
- **COCO 2017:** Lin et al., 2014. Images are CC-BY, under the
  [COCO terms of use](https://cocodataset.org/#termsofuse). `test_scene.jpg`
  is COCO val2017 image #139.
- **MPII Human Pose:** Andriluka et al., CVPR 2014. It is used via the
  [Voxel51 mirror](https://huggingface.co/datasets/Voxel51/MPII_Human_Pose_Dataset)
  and is non-commercial only.

No images are included in this repository; the scripts download them. The
trained posture weights in Releases are for **research and non-commercial use
only**, because MPII is part of their training data.

## Safety

This is a research prototype, not a certified mobility aid. Monocular
distances come from assumed object heights and an assumed FOV. YOLOv3 misses
small, transparent and low obstacles, as well as steps, kerbs and drop-offs.
Always use it alongside a cane or guide dog.
