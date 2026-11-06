## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Vargod99/assistive-navigation-vision.git
cd assistive-navigation-vision
```

### 2. Set up the environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Download model weights

```bash
python setup_weights.py
```

This downloads the YOLOv3 models, initializes the posture classifier, and fetches a test image.

### 4. Run the system

**Test with an image:**
```bash
python pipeline.py --image test_scene.jpg
```

**Run with a live webcam:**
```bash
python pipeline.py --camera 0 --speak --show
```

**Process a video:**
```bash
python pipeline.py --video walk.mp4 --save output.mp4
```

### 5. Run tests

```bash
pip install pytest
python -m pytest -q tests
```

### Requirements
- Python 3
- PyTorch
- OpenCV
- Webcam (for live navigation)
- macOS, Linux, or CUDA-compatible hardware (optional)

