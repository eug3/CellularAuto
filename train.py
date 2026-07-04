"""Training for Neural Cellular Automata.

Implements the "persistent" training regime (Experiment 2) with sample pool
and the optional damage-augmented "regenerating" regime (Experiment 3).

Key tricks reproduced faithfully from the Distill reference implementation:
  - Sample pool (default 1024 entries), per step we sample a batch of 8.
  - Sort batch by descending pre-rollout loss, replace the *highest-loss*
    sample with a fresh seed  ("reseeding trick").
  - Optional: damage the 3 lowest-loss samples by zeroing random disks.
  - Roll out for N steps sampled uniformly in [64, 96).
  - MSE loss only over the first 4 (RGBA) channels vs. premultiplied target.
  - Per-variable gradient L2 normalisation (g / (||g|| + 1e-8)).
  - Adam, lr 2e-3, drops to 2e-4 after step 2000.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn

from ca_model import CAModel
from image_utils import (CHANNEL_N, GRID_SIZE, make_circle_masks,
                         make_seed)


POOL_SIZE = 1024
BATCH_SIZE = 8
DAMAGE_N = 3                  # only used when regenerate=True
LR = 2e-3
LR_DECAY_STEP = 2000
LR_DECAY_FACTOR = 0.1
TRAIN_STEPS_LO = 64
TRAIN_STEPS_HI = 96           # exclusive -> randint(64, 96)
GRAD_EPS = 1e-8
FIRE_RATE = 0.5
DEFAULT_EPOCHS = 4000         # paper uses 8000+; 4000 is a good quick default


# --------------------------------------------------------------------------- #
# Sample pool
# --------------------------------------------------------------------------- #
@dataclass
class SamplePool:
    """A simple numpy-backed sample pool of CA states.

    State layout here is kept as NCHW numpy float32 arrays: ``[N, C, H, W]``.
    """
    states: np.ndarray
    size: int = field(init=False)

    def __post_init__(self):
        self.size = self.states.shape[0]

    def sample(self, n: int):
        idx = np.random.choice(self.size, n, replace=False)
        batch = self.states[idx].copy()
        return idx, batch

    def commit(self, idx: np.ndarray, batch: np.ndarray):
        self.states[idx] = batch

    @classmethod
    def from_seed(cls, seed: np.ndarray, size: int = POOL_SIZE) -> "SamplePool":
        return cls(states=np.repeat(seed[None, ...], size, axis=0))


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def loss_fn(states: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean-squared error over RGBA channels, all spatial positions.

    ``states``: ``[B, C, H, W]``; ``target``: ``[1, C, H, W]`` (broadcast).
    Returns a per-sample loss vector of shape ``[B]``.
    """
    rgba = states[:, :4]
    err = rgba - target[:, :4]
    # mean over (C, H, W) per sample
    return torch.mean(err * err, dim=(1, 2, 3))


def train_step(ca: CAModel,
               x0: torch.Tensor,
               target: torch.Tensor,
               optimizer: torch.optim.Optimizer,
               lr_sched: Callable[[int], float],
               grad_eps: float = GRAD_EPS,
               step_lo: int = TRAIN_STEPS_LO,
               step_hi: int = TRAIN_STEPS_HI,
               step_count: int = 0) -> tuple[torch.Tensor, float]:
    """Run one training step (a single unrolled CA rollout + gradient update).

    Returns the final states (numpy on CPU) and the mean loss of this batch.
    """
    steps = int(torch.randint(step_lo, step_hi, (1,)).item())
    x = x0

    # Build a fresh optimizer step lr via parameter group injection:
    for g in optimizer.param_groups:
        g["lr"] = lr_sched(step_count)

    optimizer.zero_grad(set_to_none=True)
    for _ in range(steps):
        x = ca(x)
    losses = loss_fn(x, target)
    loss = torch.mean(losses)
    loss.backward()

    # Per-variable gradient L2 normalisation.
    with torch.no_grad():
        for p in ca.parameters():
            if p.grad is None:
                continue
            norm = p.grad.norm()
            p.grad.copy_(p.grad / (norm + grad_eps))

    optimizer.step()

    return x.detach().cpu().numpy(), float(loss.item())


# --------------------------------------------------------------------------- #
# LR schedule
# --------------------------------------------------------------------------- #
def make_lr_schedule(decay_step: int = LR_DECAY_STEP,
                     factor: float = LR_DECAY_FACTOR) -> Callable[[int], float]:
    def lr_at(step: int) -> float:
        return LR * factor if step >= decay_step else LR
    return lr_at


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def train(target: torch.Tensor,
          ca: Optional[CAModel] = None,
          epochs: int = DEFAULT_EPOCHS,
          device: Optional[torch.device] = None,
          regenerate: bool = False,
          fire_rate: float = FIRE_RATE,
          pool_size: int = POOL_SIZE,
          batch_size: int = BATCH_SIZE,
          damage_n: int = DAMAGE_N,
          snapshot_dir: str = ".temp",
          on_progress: Optional[Callable[[int, int, float], None]] = None,
          snapshot_every: int = 200,
          save_path: Optional[str] = None,
          force_cpu: bool = True) -> CAModel:
    """Train a CAModel to grow/stabilise ``target``.

    ``target`` is a torch ``[1, CHANNEL_N, H, W]`` premultiplied tensor.
    ``on_progress(epoch, total, loss)`` is called every step (e.g. for GUI).
    Returns the trained model (located on ``device``).

    NOTE: by default ``force_cpu=True``. Backprop through the long CA rollout
    is numerically unstable on Apple's MPS backend (the trained model
    collapses to a "killer" rule), while it works perfectly on CPU. CPU is
    also plenty fast for this ~8K-parameter model. Set ``force_cpu=False``
    to attempt the requested ``device`` directly.
    """
    if force_cpu:
        train_device = torch.device("cpu")
    else:
        train_device = device or torch.device("cpu")

    if ca is None:
        ca = CAModel(fire_rate=fire_rate).to(train_device)
    else:
        ca = ca.to(train_device)
    ca.train()

    target = target.to(train_device)
    seed = make_seed(GRID_SIZE, CHANNEL_N, device=train_device)[0].cpu().numpy()    # [C,H,W]
    pool = SamplePool.from_seed(seed, size=pool_size)

    optimizer = torch.optim.Adam(ca.parameters(), lr=LR)
    lr_sched = make_lr_schedule()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    if snapshot_dir:
        os.makedirs(snapshot_dir, exist_ok=True)

    history: List[dict] = []
    t0 = time.time()

    for epoch in range(epochs):
        idx, batch_np = pool.sample(batch_size)

        # Sort by descending pre-rollout loss to find the worst sample.
        with torch.no_grad():
            pre_losses = loss_fn(torch.from_numpy(batch_np).to(train_device), target)
        order = torch.argsort(pre_losses, descending=True).cpu().numpy()
        batch_np = batch_np[order]
        idx = idx[order]

        # Reseeding trick: replace highest-loss sample with a fresh seed.
        batch_np[0] = seed

        # Optional damage: zero a random disk in the lowest-loss samples.
        if regenerate and damage_n > 0:
            masks = make_circle_masks(damage_n, GRID_SIZE, GRID_SIZE).cpu().numpy()
            batch_np[-damage_n:] = batch_np[-damage_n:] * (1.0 - masks)

        x0 = torch.from_numpy(batch_np).to(train_device)

        final, loss = train_step(ca, x0, target, optimizer, lr_sched,
                                 step_count=epoch)

        # Write evolved batch back into the pool at original indices.
        pool.commit(idx, final)

        history.append({"epoch": epoch, "loss": loss})
        if on_progress is not None:
            on_progress(epoch, epochs, loss)

        # Periodic debug snapshot.
        if snapshot_dir and (epoch % snapshot_every == 0 or epoch == epochs - 1):
            _dump_snapshot(snapshot_dir, epoch, ca, train_device)

    elapsed = time.time() - t0
    print(f"Training done in {elapsed:.1f}s. Final loss={loss:.5f}")

    metadata = {
        "losses": [h["loss"] for h in history],
        "epochs": epochs,
        "elapsed_sec": elapsed,
        "device": str(train_device),
        "regenerate": regenerate,
        "fire_rate": fire_rate,
        "pool_size": pool_size,
        "batch_size": batch_size,
    }
    if save_path:
        base = os.path.splitext(save_path)[0]
        with open(base + "_loss.json", "w") as f:
            json.dump(metadata, f, indent=2)
        # Save model weights.
        torch.save({"state_dict": ca.state_dict(),
                    "meta": {k: v for k, v in metadata.items() if k != "losses"}},
                   save_path)
        print("Saved model to", save_path)

    return ca


@torch.no_grad()
def _dump_snapshot(snapshot_dir: str, epoch: int, ca: CAModel, device):
    """Render a quick 96-step rollout of the seed and save PNG/npz to .temp."""
    from image_utils import state_to_pil
    seed = make_seed(GRID_SIZE, CHANNEL_N, device=device)
    x = seed
    ca.eval()
    for _ in range(96):
        x = ca(x)
    img = state_to_pil(x, scale=4)
    img.save(os.path.join(snapshot_dir, f"snap_{epoch:05d}.png"))
    ca.train()


# --------------------------------------------------------------------------- #
# Model (de)serialisation
# --------------------------------------------------------------------------- #
def save_model(ca: CAModel, path: str, meta: Optional[dict] = None):
    payload = {"state_dict": ca.state_dict(),
               "channel_n": ca.channel_n,
               "fire_rate": ca.fire_rate,
               "meta": meta or {}}
    torch.save(payload, path)


def load_model(path: str,
               device: Optional[torch.device] = None,
               fire_rate: float = FIRE_RATE) -> CAModel:
    payload = torch.load(path, map_location=device or "cpu", weights_only=False)
    ca = CAModel(channel_n=payload.get("channel_n", CHANNEL_N),
                 fire_rate=payload.get("fire_rate", fire_rate))
    ca.load_state_dict(payload["state_dict"])
    if device is not None:
        ca = ca.to(device)
    ca.eval()
    return ca
