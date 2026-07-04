"""Tkinter GUI for Growing Neural Cellular Automata.

Features
--------
* Open an image (tkinter file picker) -> auto loaded as the target.
* Train a CA model in a background thread (so the UI stays responsive).
* Browse & load previously-saved models from the ``models/`` directory.
* Live animation of growth/decay in the canvas.
* Play / pause / step / reset controls.
* Left-click-drag to ERASE cells -> watch the pattern regenerate (only
  robust if the model was trained with --regenerate).
* Double-click to place a new seed cell.
* Optional perception-field rotation slider (Experiment 4).
* Status bar with model/device/epoch/loss.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from typing import List, Optional

import numpy as np
import torch
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ca_model import CAModel
from image_utils import (GRID_SIZE, CHANNEL_N, get_device,
                         load_target, make_seed, state_to_pil)
from train import (DEFAULT_EPOCHS, load_model, save_model, train)

CANVAS_SCALE = 6                 # display 72x72 -> 432x432
CANVAS_SIZE = GRID_SIZE * CANVAS_SCALE


class CAApp(tk.Tk):
    """Main application window."""

    def __init__(self, initial_image: Optional[str] = None,
                 default_epochs: int = DEFAULT_EPOCHS,
                 prefer_device: str = "auto"):
        super().__init__()
        self.title("Growing Neural Cellular Automata")
        self.resizable(False, False)

        self.device = get_device(prefer_device)
        self.model: Optional[CAModel] = None
        self.target: Optional[torch.Tensor] = None
        self.target_path: Optional[str] = None
        self.state: Optional[torch.Tensor] = None

        self.playing = False
        self.angle = 0.0
        self.step_count = 0
        self.default_epochs = default_epochs

        self.train_thread: Optional[threading.Thread] = None

        self._build_ui()
        self._refresh_models_dropdown()

        if initial_image and os.path.exists(initial_image):
            self._load_image(initial_image)

        self._render_loop()

    # ----------------------------------------------------------------- UI -- #
    def _build_ui(self):
        # Left = canvas, right = controls.
        left = ttk.Frame(self, padding=8)
        left.pack(side=tk.LEFT, fill=tk.BOTH)
        right = ttk.Frame(self, padding=8)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        # Canvas
        canv_frame = ttk.Frame(left)
        canv_frame.pack()
        self.canvas = tk.Canvas(canv_frame, width=CANVAS_SIZE,
                                height=CANVAS_SIZE, bg="white",
                                highlightthickness=1, highlightbackground="#888")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self._on_erase)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)
        self._blank_image = ImageTk.PhotoImage(
            Image.new("RGB", (GRID_SIZE, GRID_SIZE), (255, 255, 255)))
        self.canvas_image = self.canvas.create_image(
            CANVAS_SIZE // 2, CANVAS_SIZE // 2, image=self._blank_image)

        legend = ttk.Label(left,
            text="Tip: left-drag to ERASE cells, double-click to PLANT a seed.")
        legend.pack(anchor=tk.W, pady=(6, 0))

        # Status bar
        self.status_var = tk.StringVar(value="Ready. Pick an image or model.")
        self.status = ttk.Label(left, textvariable=self.status_var,
                                anchor=tk.W, relief=tk.SUNKEN)
        self.status.pack(fill=tk.X, pady=(6, 0))

        # --- Right pane controls
        # Image selection
        img_frame = ttk.LabelFrame(right, text="Target image", padding=8)
        img_frame.pack(fill=tk.X)
        self.img_label = ttk.Label(img_frame, text="no image",
                                   anchor=tk.CENTER, width=22)
        self.img_label.pack()
        self.img_thumb = None  # keep ref
        ttk.Button(img_frame, text="Open image...",
                   command=self._on_open_image).pack(fill=tk.X, pady=(6, 0))

        # Training
        tr = ttk.LabelFrame(right, text="Train", padding=8)
        tr.pack(fill=tk.X, pady=8)
        ttk.Label(tr, text="Epochs:").grid(row=0, column=0, sticky=tk.W)
        self.ep_var = tk.IntVar(value=self.default_epochs)
        ttk.Spinbox(tr, from_=200, to=20000, increment=400,
                    textvariable=self.ep_var, width=7).grid(row=0, column=1)
        self.reg_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(tr, text="train with damage (regen)",
                        variable=self.reg_var).grid(row=1, column=0,
                                                    columnspan=2,
                                                    sticky=tk.W, pady=(4, 0))
        self.train_btn = ttk.Button(tr, text="Start training",
                                    command=self._on_start_training)
        self.train_btn.grid(row=2, column=0, columnspan=2,
                            sticky=tk.EW, pady=(6, 0))
        self.progress = ttk.Progressbar(tr, mode="determinate")
        self.progress.grid(row=3, column=0, columnspan=2,
                           sticky=tk.EW, pady=(6, 0))

        # Model loader
        md = ttk.LabelFrame(right, text="Saved models", padding=8)
        md.pack(fill=tk.X)
        self.model_var = tk.StringVar()
        self.model_box = ttk.Combobox(md, textvariable=self.model_var,
                                      state="readonly", width=22)
        self.model_box.pack(fill=tk.X)
        ttk.Button(md, text="Load selected model",
                   command=self._on_load_model).pack(fill=tk.X, pady=(6, 0))
        ttk.Button(md, text="Refresh list",
                   command=self._refresh_models_dropdown).pack(fill=tk.X)

        # Playback
        pb = ttk.LabelFrame(right, text="Playback", padding=8)
        pb.pack(fill=tk.X, pady=8)
        row = ttk.Frame(pb); row.pack(fill=tk.X)
        self.play_btn = ttk.Button(row, text="Play", command=self._on_play,
                                   width=6)
        self.play_btn.pack(side=tk.LEFT)
        ttk.Button(row, text="Step", command=self._on_step).pack(side=tk.LEFT)
        ttk.Button(row, text="Reset", command=self._on_reset).pack(side=tk.LEFT)

        speed_row = ttk.Frame(pb); speed_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(speed_row, text="Steps/frame:").pack(side=tk.LEFT)
        self.spf_var = tk.IntVar(value=2)
        ttk.Spinbox(speed_row, from_=1, to=10, width=4,
                    textvariable=self.spf_var).pack(side=tk.LEFT, padx=4)

        rot = ttk.Frame(pb); rot.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(rot, text="Rotate (Exp 4):").pack(side=tk.LEFT)
        self.angle_var = tk.DoubleVar(value=0.0)
        scale = ttk.Scale(rot, from_=0.0, to=6.28, orient=tk.HORIZONTAL,
                          variable=self.angle_var, length=110)
        scale.pack(side=tk.LEFT, padx=4)

        ttk.Button(right, text="Quit", command=self.destroy).pack(
            fill=tk.X, pady=(12, 0))

    # ---------------------------------------------------- helpers ------- #
    def _refresh_models_dropdown(self):
        files = []
        if os.path.isdir("models"):
            files = sorted(f for f in os.listdir("models")
                           if f.endswith(".pth"))
        self.model_box["values"] = files
        if files and not self.model_var.get():
            self.model_var.set(files[0])

    def _set_status(self, msg: str):
        self.status_var.set(msg)

    # ----------------------------------------------- image ---------- #
    def _on_open_image(self):
        path = filedialog.askopenfilename(
            title="Select target image",
            filetypes=[("Image files",
                       "*.png *.jpg *.jpeg *.webp *.bmp *.gif"),
                       ("All files", "*.*")])
        if path:
            self._load_image(path)

    def _load_image(self, path: str):
        try:
            self.target = load_target(path, device=self.device)
        except Exception as exc:
            messagebox.showerror("Image error", str(exc))
            return
        self.target_path = path
        self.img_label.config(text=os.path.basename(path))

        # Show a thumb of the target.
        from image_utils import to_rgb_display
        arr = to_rgb_display(self.target)
        thumb = Image.fromarray(arr, mode="RGB").resize((88, 88), Image.NEAREST)
        self.img_thumb = ImageTk.PhotoImage(thumb)
        self.img_label.config(image=self.img_thumb, text="",
                              compound=tk.TOP)
        self._set_status(f"Loaded image: {os.path.basename(path)}")

    # ----------------------------------------------- training ------- #
    def _on_start_training(self):
        if self.train_thread and self.train_thread.is_alive():
            messagebox.showwarning("Busy", "A training is already running.")
            return
        if self.target is None:
            messagebox.showwarning("No image", "Open a target image first.")
            return

        epochs = int(self.ep_var.get())
        regenerate = bool(self.reg_var.get())
        self.progress["maximum"] = epochs
        self.progress["value"] = 0
        self.train_btn.config(state=tk.DISABLED, text="Training...")
        self._set_status(f"Training {epochs} epochs on {self.device}...")

        base = os.path.splitext(os.path.basename(self.target_path or "ca"))[0]
        ts = time.strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join("models", f"{base}_{ts}.pth")
        os.makedirs("models", exist_ok=True)

        # Capture a copy of the target on the correct device.
        target = self.target.detach()

        def prog(epoch, total, loss):
            # Runs in the TRAINING thread; only touch Tk via thread-safe queue.
            self.after(0, lambda e=epoch, t=total, l=loss:
                       self._on_train_progress(e, t, l))

        def worker():
            model = train(target, epochs=epochs, device=self.device,
                          regenerate=regenerate, save_path=save_path,
                          snapshot_dir=".temp", on_progress=prog)
            self.after(0, lambda: self._on_train_done(model, save_path))

        self.train_thread = threading.Thread(target=worker, daemon=True)
        self.train_thread.start()

    def _on_train_progress(self, epoch, total, loss):
        self.progress["value"] = epoch + 1
        self._set_status(f"epoch {epoch+1}/{total}  loss={loss:.5f}")

    def _on_train_done(self, model: CAModel, path: str):
        self.model = model.to(self.device).eval()
        self.train_btn.config(state=tk.NORMAL, text="Start training")
        self._refresh_models_dropdown()
        # Auto-select the freshly trained model.
        self.model_var.set(os.path.basename(path))
        self._on_reset()
        self._set_status(f"Training complete. Model saved to {path}")
        messagebox.showinfo("Training complete",
                            f"Model saved to:\n{path}\n\nPress Play to grow it.")

    # ----------------------------------------------- model ---------- #
    def _on_load_model(self):
        name = self.model_var.get()
        if not name:
            messagebox.showwarning("No model", "No model selected.")
            return
        path = os.path.join("models", name)
        try:
            self.model = load_model(path, device=self.device)
        except Exception as exc:
            messagebox.showerror("Model error", str(exc))
            return
        self._on_reset()
        self._set_status(f"Loaded model: {name}")

    # ----------------------------------------------- playback ------- #
    def _on_play(self):
        if self.model is None:
            messagebox.showwarning("No model", "Train or load a model first.")
            return
        self.playing = not self.playing
        self.play_btn.config(text="Pause" if self.playing else "Play")

    def _on_step(self):
        if self.model is None:
            messagebox.showwarning("No model", "Train or load a model first.")
            return
        self._advance()

    def _on_reset(self):
        if self.model is not None:
            self.state = make_seed(GRID_SIZE, CHANNEL_N, device=self.device)
            self.step_count = 0
            self.playing = False
            self.play_btn.config(text="Play")
            self._redraw()

    # ----------------------------------------------- mouse ---------- #
    def _canvas_to_grid(self, event):
        gx = int(event.x // CANVAS_SCALE)
        gy = int(event.y // CANVAS_SCALE)
        gx = max(0, min(GRID_SIZE - 1, gx))
        gy = max(0, min(GRID_SIZE - 1, gy))
        return gx, gy

    def _on_erase(self, event):
        if self.state is None:
            return
        gx, gy = self._canvas_to_grid(event)
        radius = 5
        lo_x = max(0, gx - radius); hi_x = min(GRID_SIZE, gx + radius + 1)
        lo_y = max(0, gy - radius); hi_y = min(GRID_SIZE, gy + radius + 1)
        with torch.no_grad():
            self.state[:, :, lo_y:hi_y, lo_x:hi_x] = 0.0
        self._redraw()

    def _on_double_click(self, event):
        gx, gy = self._canvas_to_grid(event)
        with torch.no_grad():
            self.state = make_seed(GRID_SIZE, CHANNEL_N, device=self.device)
            self.state[:, :, gy, gx] = 0.0
            self.state[:, 3:, gy, gx] = 1.0
        self._redraw()

    # ----------------------------------------------- render loop --- #
    def _advance(self):
        if self.state is None:
            return
        with torch.no_grad():
            self.angle = float(self.angle_var.get())
            for _ in range(int(self.spf_var.get())):
                self.state = self.model(self.state, angle=self.angle)
                self.step_count += 1
        self._redraw()

    def _redraw(self):
        if self.state is None:
            return
        img = state_to_pil(self.state, scale=CANVAS_SCALE)
        # Keep a reference to avoid GC.
        self._tk_image = ImageTk.PhotoImage(img)
        self.canvas.itemconfig(self.canvas_image, image=self._tk_image)

    def _render_loop(self):
        if self.playing and self.model is not None:
            try:
                self._advance()
            except Exception as exc:
                self._set_status(f"error: {exc}")
                self.playing = False
                self.play_btn.config(text="Play")
        # ~60fps when playing; idle otherwise.
        self.after(30, self._render_loop)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Growing CA GUI")
    parser.add_argument("--image", help="initial target image path")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    args = parser.parse_args()

    app = CAApp(initial_image=args.image,
                default_epochs=args.epochs,
                prefer_device=args.device)
    app.mainloop()


if __name__ == "__main__":
    main()
