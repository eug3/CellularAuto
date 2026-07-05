"""Curriculum training + staged inference for the multi-morph CA.

Pipeline
--------
1. Split ``all.png`` into ``N_STAGE = 11`` target frames.
2. Train a single :class:`StageCAModel` (one-hot stage input) stage-by-stage.
   Stage ``i`` is trained only after stage ``i-1`` has converged below
   ``threshold``.  Each stage's rollout budget is recorded as ``N_i``.
3. Persist the model together with a JSON schedule file containing per-stage
   ``N_i`` (the epochs needed by the trainer to converge) and the prescribed
   inference step counts ``Steps_i = ceil(N_i * step_factor)`` (default 1.5x).
4. Staged inference: drive the CA through stages 0..N-1 with the schedule.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ca_model import CHANNEL_N, CAModel, StageCAModel
from image_utils import (CHANNEL_N, GRID_SIZE, make_circle_masks, make_seed,
                         state_to_pil)
from train import (BATCH_SIZE, DAMAGE_N, FIRE_RATE, GRAD_EPS, LR,
                   LR_DECAY_FACTOR, LR_DECAY_STEP, POOL_SIZE,
                   TRAIN_STEPS_HI, TRAIN_STEPS_LO, SamplePool, loss_fn,
                   make_lr_schedule)

N_STAGE = 11
DEFAULT_THRESHOLD = 5e-3          # mean target MSE before moving to next stage
DEFAULT_STAGE_MAX_EPOCHS = 4000   # safety cap per stage
DEFAULT_STEP_FACTOR = 1.5         # 1.5x extra steady-state steps at inference
DEFAULT_SAVE_NAME = "staged"
SCHEDULE_NAME = "stage_schedule.json"


# --------------------------------------------------------------------------- #
# Training step (stage-conditioned rollout)
# --------------------------------------------------------------------------- #
def stage_train_step(ca: StageCAModel,
                     x0: torch.Tensor,
                     target: torch.Tensor,
                     stage: int,
                     optimizer: torch.optim.Optimizer,
                     grad_eps: float = GRAD_EPS,
                     step_lo: int = TRAIN_STEPS_LO,
                     step_hi: int = TRAIN_STEPS_HI) -> tuple[np.ndarray, float]:
    """One unrolled rollout + gradient update for a single stage."""
    steps = int(torch.randint(step_lo, step_hi, (1,)).item())
    optimizer.zero_grad(set_to_none=True)
    x = x0
    for _ in range(steps):
        x = ca(x, stage=stage)
    losses = loss_fn(x, target)                   # [B]
    loss = torch.mean(losses)
    loss.backward()

    with torch.no_grad():
        for p in ca.parameters():
            if p.grad is None:
                continue
            norm = p.grad.norm()
            p.grad.copy_(p.grad / (norm + grad_eps))
    optimizer.step()
    return x.detach().cpu().numpy(), float(loss.item())


# --------------------------------------------------------------------------- #
# Curriculum loop
# --------------------------------------------------------------------------- #
@dataclass
class StageRecord:
    stage: int
    converged_epochs: int = 0       # epochs spent in this stage (0 if not done)
    final_loss: float = float("inf")
    status: str = "pending"          # pending | converged | capped | active


@dataclass
class Schedule:
    n_stage: int
    step_factor: float
    rollout_lo: int
    rollout_hi: int
    threshold: float
    stages: List[StageRecord] = field(default_factory=list)
    elapsed_sec: float = 0.0

    def inference_steps(self, stage: int, fallback: int = TRAIN_STEPS_HI) -> int:
        """Inference budget: 1.5 * convergence_epochs (rounded).
        Falls back to ``fallback`` if the stage never converged.
        """
        rec = self.stages[stage]
        n = rec.converged_epochs if rec.converged_epochs > 0 else fallback
        return int(math.ceil(n * self.step_factor))

    def to_json(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = asdict(self)
        data["inference_steps"] = [self.inference_steps(i)
                                   for i in range(self.n_stage)]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def train_staged(targets: torch.Tensor,
                 ca: Optional[StageCAModel] = None,
                 n_stages: int = N_STAGE,
                 epochs_per_stage: int = DEFAULT_STAGE_MAX_EPOCHS,
                 threshold: float = DEFAULT_THRESHOLD,
                 step_factor: float = DEFAULT_STEP_FACTOR,
                 device: Optional[torch.device] = None,
                 regenerate: bool = False,
                 fire_rate: float = FIRE_RATE,
                 pool_size: int = POOL_SIZE,
                 batch_size: int = BATCH_SIZE,
                 damage_n: int = DAMAGE_N,
                 snapshot_dir: Optional[str] = ".temp",
                 snapshot_every: int = 200,
                 save_path: Optional[str] = None,
                 schedule_path: Optional[str] = None,
                 on_progress: Optional[Callable[[int, int, int, float], None]] = None,
                 on_snapshot: Optional[Callable[[int, int, "np.ndarray"], None]] = None,
                 snapshot_preview_every: int = 25,
                 snapshot_preview_steps: int = TRAIN_STEPS_HI,
                 force_cpu: bool = True) -> tuple[StageCAModel, Schedule]:
    """Curriculum-train a :class:`StageCAModel`.

    ``targets``: ``[n_stages, CHANNEL_N, H, W]``.

    ``on_progress(stage, stage_epoch, total_epoch, loss)`` is called after
    each successful optimizer step.

    ``on_snapshot(stage, total_epoch, rgb_array)`` is called periodically
    (every ``snapshot_preview_every`` total epochs) with an HxWx3 uint8
    preview of the current stage's growth, useful for a live GUI.
    """
    if force_cpu:
        dev = torch.device("cpu")
    else:
        dev = device or torch.device("cpu")

    if ca is None:
        ca = StageCAModel(channel_n=CHANNEL_N,
                          stage_n=n_stages,
                          fire_rate=fire_rate).to(dev)
    else:
        ca = ca.to(dev)
    ca.train()

    targets = targets.to(dev)
    if targets.dim() == 4 and targets.shape[0] != n_stages:
        raise ValueError(f"targets has {targets.shape[0]} frames, expected {n_stages}")

    optimizer = torch.optim.Adam(ca.parameters(), lr=LR)
    lr_sched = make_lr_schedule()

    schedule = Schedule(n_stage=n_stages,
                        step_factor=step_factor,
                        rollout_lo=TRAIN_STEPS_LO,
                        rollout_hi=TRAIN_STEPS_HI,
                        threshold=threshold,
                        stages=[StageRecord(stage=i) for i in range(n_stages)])

    # A single shared sample pool seeded from the centre seed.  When moving
    # to a new stage we keep the pool (so morphology persists) but only
    # optimise toward the new target.
    seed = make_seed(GRID_SIZE, CHANNEL_N, device=dev)[0].cpu().numpy()
    pool = SamplePool.from_seed(seed, size=pool_size)

    if snapshot_dir:
        os.makedirs(snapshot_dir, exist_ok=True)

    t0 = time.time()
    total_epoch = 0

    for stage in range(n_stages):
        target = targets[stage:stage + 1]            # [1, C, H, W]
        record = schedule.stages[stage]
        record.status = "active"
        stage_epoch = 0
        last_loss = float("inf")

        while stage_epoch < epochs_per_stage:
            idx, batch_np = pool.sample(batch_size)

            # Sort by descending pre-rollout loss (worst first).
            with torch.no_grad():
                pre_losses = loss_fn(torch.from_numpy(batch_np).to(dev), target)
            order = torch.argsort(pre_losses, descending=True).cpu().numpy()
            batch_np = batch_np[order]
            idx = idx[order]

            # Replace worst sample with a fresh seed (reseeding trick).
            batch_np[0] = seed

            if regenerate and damage_n > 0:
                masks = make_circle_masks(damage_n, GRID_SIZE, GRID_SIZE).cpu().numpy()
                batch_np[-damage_n:] = batch_np[-damage_n:] * (1.0 - masks)

            x0 = torch.from_numpy(batch_np).to(dev)

            for g in optimizer.param_groups:
                g["lr"] = lr_sched(total_epoch)

            final, loss = stage_train_step(ca, x0, target, stage, optimizer)
            pool.commit(idx, final)

            stage_epoch += 1
            total_epoch += 1
            last_loss = loss

            if on_progress is not None:
                on_progress(stage, stage_epoch, total_epoch, loss)

            if snapshot_dir and (total_epoch % snapshot_every == 0):
                _dump_stage_snapshot(snapshot_dir, total_epoch, ca, stage, dev)

            if on_snapshot is not None and (total_epoch % snapshot_preview_every == 0):
                rgb = render_stage_preview(ca, stage, dev,
                                           steps=snapshot_preview_steps,
                                           scale=1)
                on_snapshot(stage, total_epoch, rgb)

            # Convergence check (smoothed over a short window would be nicer,
            # but the simple immediate check is faithful to the spec).
            if loss < threshold:
                record.converged_epochs = stage_epoch
                record.final_loss = loss
                record.status = "converged"
                print(f"[stage {stage}] converged at epoch {stage_epoch} "
                      f"(loss={loss:.5f} < {threshold})")
                break
        else:
            record.converged_epochs = stage_epoch
            record.final_loss = last_loss
            record.status = "capped"
            print(f"[stage {stage}] reached budget without convergence "
                  f"(final loss={last_loss:.5f}); recording N_{stage} "
                  f"= {stage_epoch} as a fallback.")

        # Persist intermediate snapshot of BOTH model + schedule so a long
        # run can be inspected midway.
        if save_path:
            save_staged_model(ca, save_path, schedule)
        if schedule_path:
            schedule.to_json(schedule_path)
        if snapshot_dir:
            _dump_stage_snapshot(snapshot_dir, total_epoch, ca, stage, dev,
                                 final=True)

    schedule.elapsed_sec = time.time() - t0
    print(f"Curriculum done in {schedule.elapsed_sec:.1f}s "
          f"({sum(s.converged_epochs for s in schedule.stages)} total epochs).")

    if save_path:
        save_staged_model(ca, save_path, schedule)
    if schedule_path:
        schedule.to_json(schedule_path)

    return ca, schedule


# --------------------------------------------------------------------------- #
# Snapshots
# --------------------------------------------------------------------------- #
@torch.no_grad()
def render_stage_preview(ca: StageCAModel,
                         stage: int,
                         device,
                         steps: int = TRAIN_STEPS_HI,
                         scale: int = 1) -> np.ndarray:
    """Render the current model's growth on a fresh seed for ``stage``.

    Returns an HxWx3 uint8 RGB array (already upscaled by ``scale``).
    Safe to call while the model is in training mode - it temporarily
    switches to eval and restores the previous mode.
    """
    from image_utils import to_rgb_display
    seed = make_seed(GRID_SIZE, CHANNEL_N, device=device)
    x = seed
    was_training = ca.training
    ca.eval()
    for _ in range(steps):
        x = ca(x, stage=stage)
    arr = to_rgb_display(x)                       # HxWx3 uint8
    if was_training:
        ca.train()
    if scale != 1:
        from PIL import Image as _PILImage
        h, w, _ = arr.shape
        arr = np.asarray(
            _PILImage.fromarray(arr, mode="RGB").resize(
                (w * scale, h * scale), _PILImage.NEAREST))
    return arr


@torch.no_grad()
def _dump_stage_snapshot(snapshot_dir: str,
                         global_epoch: int,
                         ca: StageCAModel,
                         stage: int,
                         device,
                         final: bool = False):
    arr = render_stage_preview(ca, stage, device, scale=4)
    img = Image.fromarray(arr, mode="RGB")
    tag = "final" if final else "snap"
    img.save(os.path.join(snapshot_dir, f"stage{stage:02d}_{tag}_{global_epoch:06d}.png"))


# --------------------------------------------------------------------------- #
# (De)serialisation
# --------------------------------------------------------------------------- #
def save_staged_model(ca: StageCAModel,
                      path: str,
                      schedule: Optional[Schedule] = None):
    payload = {
        "state_dict": ca.state_dict(),
        "channel_n": ca.channel_n,
        "stage_n": ca.stage_n,
        "fire_rate": ca.fire_rate,
        "schedule": asdict(schedule) if schedule else None,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)
    print("Saved staged model to", path)


def load_staged_model(path: str,
                      device: Optional[torch.device] = None,
                      fire_rate: float = FIRE_RATE) -> StageCAModel:
    payload = torch.load(path, map_location=device or "cpu", weights_only=False)
    ca = StageCAModel(channel_n=payload.get("channel_n", CHANNEL_N),
                      stage_n=payload.get("stage_n", N_STAGE),
                      fire_rate=payload.get("fire_rate", fire_rate))
    ca.load_state_dict(payload["state_dict"])
    if device is not None:
        ca = ca.to(device)
    ca.eval()
    return ca


def load_schedule(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Staged inference pipeline
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_pipeline(ca: StageCAModel,
                 schedule_or_steps,
                 device: Optional[torch.device] = None,
                 out_dir: Optional[str] = None,
                 frame_every: int = 4,
                 extra_steps_per_stage: int = 0) -> List[np.ndarray]:
    """Drive the CA through all stages in order and collect frames.

    ``schedule_or_steps`` may be either:
      * a :class:`Schedule` instance, or
      * a list/tuple of integers (per-stage step counts).

    Returns a list of HxWx3 uint8 RGB frames.
    """
    dev = device or next(ca.parameters()).device
    ca.eval()

    if isinstance(schedule_or_steps, Schedule):
        steps_list = [schedule_or_steps.inference_steps(i)
                      for i in range(schedule_or_steps.n_stage)]
        n_stage = schedule_or_steps.n_stage
    else:
        steps_list = list(schedule_or_steps)
        n_stage = len(steps_list)

    x = make_seed(GRID_SIZE, CHANNEL_N, device=dev)
    frames: List[np.ndarray] = []

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    frame_idx = 0

    for stage in range(n_stage):
        steps = steps_list[stage] + extra_steps_per_stage
        for s in range(steps):
            x = ca(x, stage=stage)
            if s % frame_every == 0 or s == steps - 1:
                frame = _state_to_frame(x)
                frames.append(frame)
                if out_dir:
                    from PIL import Image as _PILImage
                    _PILImage.fromarray(frame).save(
                        os.path.join(out_dir, f"f_{frame_idx:05d}_s{stage:02d}.png"))
                frame_idx += 1
    return frames


def _state_to_frame(state: torch.Tensor) -> np.ndarray:
    """[1,C,H,W] -> HxWx3 uint8."""
    from image_utils import to_rgb_display
    return to_rgb_display(state)
