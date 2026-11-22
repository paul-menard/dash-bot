# Geometry Dash AI Bot

A Python bot that plays **Geometry Dash** on its own. It reads the game window in real time, detects the player and the obstacles with a custom-trained **YOLO** model, and sends keyboard inputs to jump or fly.

It handles two game modes, each with its own logic:

- **Cube**: rule-based jump decisions from the distance to upcoming obstacles.
- **Rocket**: a cost-map navigation system with a predictive "gas" controller (open/close thrust).

> **Platform:** Windows only (uses `win32gui` and `pydirectinput`) with an **NVIDIA GPU** (CUDA).

---

## How it works

```
Game window ──► capture.py ──► YOLO (best.pt) ──► logic.py ──► controls.py ──► Game
   (mss)         screenshot      detections       decisions      key presses
```

1. **Capture** (`capture.py`): finds the `Geometry Dash` window, grabs the client area with `mss`, and returns a BGR NumPy frame. The process is set to per-monitor DPI awareness so coordinates match physical pixels.
2. **Detection** (`main.py`): the frame goes through the YOLO model (`conf=0.55`). The output boxes and class IDs are kept as GPU tensors.
3. **Decision** (`logic.py`): an OOP design with a base `Character` class and two subclasses, `Cube` and `Rocket`. The bot swaps character class automatically when the detected player type changes.
4. **Control** (`controls.py`): sends key presses (`space`) through `pydirectinput`, using a small thread pool so input never blocks the vision loop.

### Detection classes

| ID | Class | Role |
|----|-------|------|
| 2 | Cube | Player (cube mode) |
| 5 | Rocket | Player (rocket mode) |
| 0, 4, 7 | Obstacles | Used for decisions |

### Cube logic

`Cube.is_jump_safe` projects a danger zone in front of the cube and checks, with tensor ops, whether any obstacle box overlaps it. Jumps are triggered when an obstacle gets within a distance threshold.

### Rocket logic

For each frame, `Rocket.new_navigation` builds a 1D **cost map** along the screen height:

- obstacle interiors get a very high cost, weighted by urgency (closer = more urgent);
- a parabolic **repulsion field** is added around each obstacle, which centers the rocket in gaps;
- with no obstacle in sight, a light pull toward the vertical center prevents drifting.

The target is the `argmin` of the cost map, smoothed with an EMA. The controller then predicts when to open or close the thrust, using the rocket's physical constants (gravity, thrust) and asymmetric damping.

A live **heatmap window** of the cost map is shown next to the game while in rocket mode, which is handy for debugging.

---

## Repository structure

```
.
├── main.py                      # entry point: capture → detect → decide → display
├── src/
│   ├── capture.py               # window capture (mss + win32gui)
│   ├── logic.py                 # Character / Cube / Rocket behavior
│   └── controls.py              # keyboard inputs (pydirectinput)
├── runs/
│   └── detect/
│       └── train/
│           └── weights/
│               └── best.pt      # trained YOLO weights (included in the repo)
├── requirements.txt
└── README.md
```

> `main.py` adds `./src` to `sys.path`, so the three modules live in `src/`.

---

## Installation

### Requirements

- Windows 10/11
- Python 3.10+
- NVIDIA GPU with a recent driver
- Geometry Dash (Steam), running in **windowed mode**

### Setup

```bash
git clone <your-repo-url>
cd <your-repo-name>

python -m venv .venv
.venv\Scripts\activate

# Install PyTorch with CUDA first (pick the right CUDA version on https://pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install ultralytics opencv-python numpy mss pywin32 pydirectinput
```

Check that CUDA is available:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Model weights

The trained weights are included in the repository at:

```
runs/detect/train/weights/best.pt
```

No extra download is needed. `main.py` loads them with:

```python
model = YOLO('runs/detect/train/weights/best.pt')
```

---

## Usage

1. Launch Geometry Dash in windowed mode, with the window titled exactly `Geometry Dash`.
2. Start a level.
3. Run the bot:

```bash
python main.py
```

4. Two windows appear: the annotated detection view (`Computer Vision`) and, in rocket mode, the cost-map heatmap.
5. Press **`q`** in the detection window to quit.

---

## Configuration and known limits

- **Window title:** `WindowCapture('Geometry Dash')` in `main.py`.
- **Window offsets:** `OFFSET_X` / `OFFSET_Y` in `capture.py` crop the window borders and title bar. Adjust them if your Windows theme or scaling differs.
- **Detection threshold:** `conf=0.55` in `main.py`.
- **Physics constants:** the Rocket's `thrust`, `gravity`, and the distance thresholds are tuned for one specific resolution and speed. Expect to re-tune them for other setups.
- **CUDA device:** `cuda:0` is hardcoded in `main.py` and `logic.py`. There is no CPU fallback.

---

## Tech stack

Python · PyTorch (CUDA) · Ultralytics YOLO · OpenCV · mss · pywin32 · pydirectinput

---

## Disclaimer

This project is for educational and research purposes (computer vision and real-time control). It is not affiliated with RobTop Games.
