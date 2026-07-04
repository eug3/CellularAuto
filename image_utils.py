"""Image / tensor utilities for Growing Neural Cellular Automata.

Layout convention: PyTorch NCHW.
State tensors have shape ``[B, CHANNEL_N, H, W]`` with channel ordering
    0:R, 1:G, 2:B, 3:Alpha, 4..15: hidden channels.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

CHANNEL_N = 16
TARGET_SIZE = 40
TARGET_PADDING = 16              # 40 + 2*16 = 72
GRID_SIZE = TARGET_SIZE + 2 * TARGET_PADDING   # 72


# --------------------------------------------------------------------------- #
# Device selection
# --------------------------------------------------------------------------- #
def get_device(prefer: str = "auto") -> torch.device:
    """Pick a torch device.

    ``prefer``: "auto" | "mps" | "cuda" | "cpu".
    Falls back gracefully: requested -> mps -> cuda -> cpu.
    """
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if prefer == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Target image loading
# --------------------------------------------------------------------------- #
def _fit_into(img: Image.Image, size: int) -> Image.Image:
    """Scale image (keep aspect) and center it on a transparent ``size x size`` canvas."""
    thumb = img.copy()
    thumb.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    off_x = (size - thumb.width) // 2
    off_y = (size - thumb.height) // 2
    canvas.paste(thumb, (off_x, off_y), thumb)
    return canvas


def load_target(image_path: str,
                target_size: int = TARGET_SIZE,
                device=None) -> torch.Tensor:
    """Load any image, scale to ``target_size`` and pad to ``GRID_SIZE``.

    Returns a tensor of shape ``[1, CHANNEL_N, GRID_SIZE, GRID_SIZE]``.
    Channels 0..3 hold RGBA (RGB premultiplied by alpha); 4..15 are zero.
    """
    img = Image.open(image_path).convert("RGBA")
    img = _fit_into(img, target_size)

    arr = np.asarray(img, dtype=np.float32) / 255.0    # HxWx4
    rgb = arr[..., :3]
    alpha = arr[..., 3:4]
    rgb = rgb * alpha                                   # premultiply alpha
    rgba = np.concatenate([rgb, alpha], axis=-1)

    p = TARGET_PADDING
    g = GRID_SIZE
    grid = np.zeros((g, g, CHANNEL_N), dtype=np.float32)
    grid[p:p + target_size, p:p + target_size, :4] = rgba

    t = torch.from_numpy(grid).permute(2, 0, 1).unsqueeze(0).contiguous()
    if device is not None:
        t = t.to(device)
    return t


# --------------------------------------------------------------------------- #
# Seed & display
# --------------------------------------------------------------------------- #
def make_seed(size: int = GRID_SIZE,
              channel_n: int = CHANNEL_N,
              batch: int = 1,
              device=None) -> torch.Tensor:
    """Single seed cell at the grid centre.

    All channels of the centre cell except RGB are set to 1.0; everything
    else is zero. Shape ``[batch, channel_n, size, size]``.
    """
    x = torch.zeros((batch, channel_n, size, size), dtype=torch.float32)
    x[:, 3:, size // 2, size // 2] = 1.0
    if device is not None:
        x = x.to(device)
    return x


def to_rgb_display(state: torch.Tensor) -> np.ndarray:
    """Convert an RGBA-premultiplied state tensor to a displayable RGB uint8 image.

    Input: ``[B, >=4, H, W]`` or ``[>=4, H, W]``. Returns ``[H, W, 3]`` uint8
    using the first batch element if batched.
    """
    if state.dim() == 4:
        state = state[0]
    state = state.detach().cpu().float()
    rgb = state[:3]
    alpha = state[3:4]
    out = 1.0 - alpha + rgb                            # un-premultiply
    out = np.clip(out.numpy(), 0.0, 1.0)
    out = (out * 255).astype(np.uint8)
    return np.transpose(out, (1, 2, 0))                # HWC


def state_to_pil(state: torch.Tensor, scale: int = 6) -> Image.Image:
    """Render a CA state to a (scaled) PIL RGB image for tkinter."""
    arr = to_rgb_display(state)
    img = Image.fromarray(arr, mode="RGB")
    if scale != 1:
        img = img.resize((arr.shape[1] * scale, arr.shape[0] * scale),
                         Image.NEAREST)
    return img


# --------------------------------------------------------------------------- #
# Damage masks (Experiment 3)
# --------------------------------------------------------------------------- #
def make_circle_masks(n: int, h: int, w: int, device=None) -> torch.Tensor:
    """``n`` circular binary masks of shape ``[n, 1, h, w]``.

    Used to zero out (damage) random disk-shaped regions of pool samples
    during regeneration training.
    """
    xs = torch.linspace(-1.0, 1.0, w, device=device).view(1, 1, 1, w)
    ys = torch.linspace(-1.0, 1.0, h, device=device).view(1, 1, h, 1)
    center = torch.rand(2, n, 1, 1, device=device) * 2 - 1   # [-1, 1]
    center = center * 0.5                                    # +/-0.5 jitter
    radius = torch.rand(n, 1, 1, 1, device=device) * 0.3 + 0.1   # [0.1, 0.4]
    dx = (xs - center[0].view(n, 1, 1, 1)) / radius
    dy = (ys - center[1].view(n, 1, 1, 1)) / radius
    mask = (dx * dx + dy * dy) < 1.0
    return mask.float()                                     # [n,1,h,w]
