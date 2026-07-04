"""Neural Cellular Automata model (PyTorch port of the Distill paper).

Architecture (faithful to Mordvintsev et al. 2020):

    state [B, 16, H, W]
        |
        +-- perceive:  depthwise 3x3 conv, kernels = [identity, sobel_x, sobel_y]
        |        (optionally rotated for Experiment 4)
        v
    perception [B, 48, H, W]
        |
        +-- UpdateCNN: Conv2d(48->128, 1) + ReLU -> Conv2d(128->16, 1)
        |              last layer weights initialised to ZERO (do-nothing init)
        v
    delta [B, 16, H, W]   (stochastic update mask, fire_rate=0.5)
        |
        +-- combine with alive masking (pre & post, alpha>0.1)
        v
    new_state [B, 16, H, W]

~8.3K parameters total.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CHANNEL_N = 16


def _build_perception_kernel(channel_n: int, angle: float = 0.0) -> torch.Tensor:
    """Return a depthwise-conv kernel of shape ``[channel_n*3, 1, 3, 3]``.

    For each input channel it produces 3 outputs stacked in this order:
    identity, dx, dy. The dx/dy kernels can be rotated by ``angle`` radians
    (Experiment 4: rotating the perceptive field).
    """
    identify = np.outer([0, 1, 0], [0, 1, 0]).astype(np.float32)
    dx = (np.outer([1, 2, 1], [-1, 0, 1]).astype(np.float32)
          * np.float32(1.0 / 8.0))                       # Sobel x, float32
    dy = dx.T                                            # Sobel y

    c, s = np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)
    dx_r = c * dx - s * dy
    dy_r = s * dx + c * dy

    # Per channel, the 3 stacked kernels.
    kernels = np.stack([identify, dx_r, dy_r], axis=-1)   # [3, 3, 3]      (h,w,k)
    kernels = np.tile(kernels[None, ...], (channel_n, 1, 1, 1))  # [C,3,3,3]
    kernels = np.transpose(kernels, (0, 3, 1, 2))         # [C, 3, 3, 3]
    kernels = kernels.reshape(channel_n * 3, 1, 3, 3)
    return torch.from_numpy(kernels)


def perceive(x: torch.Tensor, angle: float = 0.0) -> torch.Tensor:
    """Compute the perception vector by depthwise 3x3 convolution.

    Input  ``[B, C, H, W]`` -> output ``[B, 3C, H, W]``.
    """
    b, c, h, w = x.shape
    kernel = _build_perception_kernel(c, angle).to(x.device).to(x.dtype)
    return F.conv2d(x, kernel, padding=1, groups=c)


def get_living_mask(x: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
    """Boolean mask of living cells: 3x3 max-pool of alpha > threshold.

    Returns ``[B, 1, H, W]`` float32 (1.0 = alive).
    """
    alpha = x[:, 3:4]
    pooled = F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1)
    return (pooled > threshold).float()


class UpdateCNN(nn.Module):
    """The learnable cell-update network: 1x1 conv 48->128 (ReLU) -> 1x1 conv 128->16.

    Last-layer weights are zero-initialised so the model output is zero at
    init (do-nothing behaviour -> seed stays unchanged).
    """

    def __init__(self, channel_n: int = CHANNEL_N, hidden_n: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_n * 3, hidden_n, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden_n, channel_n, kernel_size=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, perception: torch.Tensor) -> torch.Tensor:
        x = self.conv1(perception)
        x = F.relu(x)
        x = self.conv2(x)
        return x


class CAModel(nn.Module):
    """Full Neural CA model wrapper: perceive -> UpdateCNN -> mask."""

    def __init__(self, channel_n: int = CHANNEL_N, fire_rate: float = 0.5):
        super().__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate
        self.update_net = UpdateCNN(channel_n)

    def forward(self,
                x: torch.Tensor,
                fire_rate: Optional[float] = None,
                angle: float = 0.0,
                step_size: float = 1.0) -> torch.Tensor:
        if fire_rate is None:
            fire_rate = self.fire_rate

        pre = get_living_mask(x)

        y = perceive(x, angle)
        dx = self.update_net(y) * step_size

        # Stochastic per-cell update mask, broadcast across channels.
        mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3],
                           device=x.device, dtype=x.dtype) <= fire_rate).float()
        x = x + dx * mask

        post = get_living_mask(x)
        life = pre * post
        return x * life

    @torch.no_grad()
    def step(self, x: torch.Tensor, angle: float = 0.0) -> torch.Tensor:
        """Convenience inference step (no grad, default fire rate)."""
        return self.forward(x, angle=angle)
