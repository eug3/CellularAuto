# Growing Neural Cellular Automata (PyTorch)

A faithful PyTorch reproduction of **Mordvintsev, Randazzo, Niklasson & Levin
(2020), _"Growing Neural Cellular Automata"_** ([Distill](https://distill.pub/2020/growing-ca/)).

Each cell of a 2D grid carries a 16-channel state (RGBA + 12 hidden). A tiny
learnable "genome" (~8 300 parameters) — a 2-layer 1×1 conv network — describes
how each cell updates itself from its 3×3 neighbourhood. Trained
end-to-end with backprop-through-time, this rule learns to **grow a target
image from a single seed pixel**, **stabilise it** (persistence attractor),
and — if trained with damage — **regenerate** the pattern after being erased.

This package ships a CLI (`main.py`) and a self-contained **tkinter GUI** for
training and watching growth live. No web, no notebook, just Python + the
standard library.

---

## Features
- **Train on any image** you supply (PNG/JPG/WEBP/BMP/GIF): emoji, logos,
  doodles, etc.
- **Live tkinter viewer**: play / pause / step / reset, plus a perception
  rotation slider (Experiment 4 — produces rotated copies of the pattern
  without re-training).
- **Interactive regeneration**: left-drag on the canvas to erase cells and
  watch the pattern repair itself (use the *regen* training option).
- **Sample-pool training** (Experiment 2, the canonical Growing-CA result).
- **Damage-augmented training** (Experiment 3) for true regeneration.
- **Apple Silicon MPS or CPU** for rendering; CPU training (numerically
  stable on this small model — see *Notes*).

---

## Setup

```bash
cd CellularAuto
python3.12 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Requires **Python 3.12** (or any 3.x with torch wheels). Python 3.13/3.14
> are currently ahead of PyTorch wheel availability. The project depends only
> on `torch`, `numpy`, and `Pillow`; `tkinter` ships with CPython.

Sample target images are bundled in `targets/` (heart, smiley, lizard, ring).

---

## Quick start

### 1. Open the GUI
```bash
python main.py
```
Then:
1. Click **Open image…** and pick e.g. `targets/heart.png`.
2. (Optional) tick **train with damage (regen)** if you want regeneration.
3. Click **Start training**. Training runs in a background thread; the
   progress bar shows the epoch. With the default 4000 epochs on CPU it
   takes ~10–15 min and the result is a well-formed pattern.
4. When it finishes, **Play** will animate the pattern growing from the
   central seed.

### 2. Train from the CLI (no GUI)
```bash
python main.py --image targets/heart.png --train --epochs 4000
# weights -> models/heart.pth, loss curve -> models/heart_loss.json
```
Add `--regen` to train with random damage (Experiment 3).

### 3. Play a trained model headless (dumps PNG frames)
```bash
python main.py --image targets/heart.png --play \
               --model models/heart.pth --steps 96 --out out_frames
```

---

## How it works

```
state [B, 16, 72, 72]
   │ perceive: depthwise 3×3 conv with [identity, Sobel-x, Sobel-y] kernels
   ▼
perception [B, 48, 72, 72]
   │ UpdateCNN : Conv 1×1 (48→128) + ReLU → Conv 1×1 (128→16)
   │           (last-layer weights init = 0 → "do-nothing" at start)
   ▼
update Δ [B, 16, 72, 72]   × per-cell stochastic mask  (fire_rate = 0.5)
   │ +living masking   (a cell is alive iff max-pool₃ₓ₃(α) > 0.1, before
   │                   AND after the update)
   ▼
new state
```

Training (Persistent, Experiment 2):
- Pool of 1024 starting states, initialised to a single seed.
- Per step: sample a batch of 8, sort by descending pre-rollout loss,
  **replace the highest-loss sample with a fresh seed** ("reseeding trick").
- Roll out for a random number of steps in `[64, 96)`.
- Loss = MSE over the RGBA channels vs. the (alpha-premultiplied) target.
- Per-variable gradient L2 normalisation: `g ← g / (‖g‖ + 1e-8)`.
- Adam, `lr = 2e-3`, drops to `2e-4` after epoch 2000.

Regeneration (Experiment 3): additionally zero a random disk of radius
`r ∈ [0.1, 0.4]` (in normalised grid coords) in the 3 lowest-loss pool
samples each step. The learned dynamics then forms a wide attractor around
the target and recovers from arbitrary damage.

Rotation (Experiment 4): the Sobel-x / Sobel-y perception kernels are
rotated by an angle `θ` before each conv. The same trained weights then
grow the pattern at any chosen orientation — no re-training needed. Use the
slider in the GUI.

---

## Choice of device

|              | training                  | rendering (GUI / --play)        |
|--------------|---------------------------|---------------------------------|
| Apple Silicon| **CPU** (default, stable) | MPS (fast) or CPU               |
| NVIDIA       | CUDA                      | CUDA                            |
| Other        | CPU                       | CPU                             |

**Why CPU training on Apple Silicon?** Back-propagating through the long
`[64, 96)`-step CA rollout is numerically unstable on the MPS backend with
current PyTorch (the trained model collapses into a "killer" rule that
vanishes the seed). The model is small (~8 K params) so CPU training is
both stable and only a few minutes per 4000 epochs.

You can override this with the `--device` flag (`auto` / `mps` / `cuda` /
`cpu`) — it controls *rendering*; to attempt MPS training anyway, edit the
`force_cpu=True` argument passed to `train()` in `main.py` / `gui.py`.

---

## File layout
```
CellularAuto/
├── ca_model.py     # CAModel: perceive + UpdateCNN + alive mask + stochastic update
├── image_utils.py  # target loading, seed, display, damage masks, device picker
├── train.py        # SamplePool, train_step, training loop, save/load
├── gui.py          # tkinter app: viewer + train thread + erase + rotation
├── main.py         # CLI entry (GUI / --train / --play)
├── requirements.txt
├── targets/        # bundled sample images
├── models/         # trained .pth weights land here
└── .temp/          # debug snapshots & scratch (gitignored)
```

---

## Tips & troubleshooting
- **Best images**: simple, iconic shapes with limited colour palette and
  generous transparent margins — emoji, logos, cartoon characters. A photo
  will rarely converge cleanly.
- **Loss not dropping / pattern doesn't grow**: increase `--epochs`. The
  canonical paper uses 8000. Anything ≥2000 generally looks good.
- **Pattern grows then explodes**: classic Growing-CA instability. Use the
  persistent pool training (default) and train longer; consider lowering
  fire-rate noise or training without `--regen`.
- **Click-and-drag erasing has no effect**: the loaded model wasn't trained
  with damage. Retrain with the *regen* box checked (or `--regen`).
- **Installed Python is 3.14 and `pip install torch` fails**: use 3.12, e.g.
  `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`.

---

## References
- Mordvintsev, Randazzo, Niklasson, Levin. *Growing Neural Cellular Automata*.
  Distill, 2020. DOI [10.23915/distill.00023](https://doi.org/10.23915/distill.00023).
- Reference TF notebook: <https://github.com/google-research/self-organising-systems/blob/master/notebooks/growing_ca.ipynb>
