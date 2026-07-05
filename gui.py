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
                         load_target, load_stage_targets, make_seed,
                         state_to_pil)
from train import (DEFAULT_EPOCHS, load_model, save_model, train)

CANVAS_SCALE = 2                 # display 224x224 -> 448x448 (kept ~same size)
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

        self.playing = False
        self.angle = 0.0
        self.step_count = 0
        self.default_epochs = default_epochs

        self.train_thread: Optional[threading.Thread] = None

        # Single-tab state.
        self.model: Optional[CAModel] = None
        self.target: Optional[torch.Tensor] = None
        self.target_path: Optional[str] = None
        self.state: Optional[torch.Tensor] = None

        # Staged-curriculum state.
        self.stage_targets: Optional[torch.Tensor] = None   # [N, C, H, W]
        self.stage_sheet_path: Optional[str] = None
        self.stage_model = None           # currently loaded StageCAModel
        self._staged_schedule = None      # cached Schedule for default steps
        self.staged_playing = False
        self.staged_state: Optional[torch.Tensor] = None

        # Which tab owns the shared canvas; updated by _on_tab_changed.
        self.active_tab = "single"

        self._build_ui()
        self._refresh_models_dropdown()

        if initial_image and os.path.exists(initial_image):
            self._load_image(initial_image)

        self._render_loop()

    # ----------------------------------------------------------------- UI -- #
    def _build_ui(self):
        # Left = canvas, right = controls (the controls hold the Notebook).
        left = ttk.Frame(self, padding=8)
        left.pack(side=tk.LEFT, fill=tk.BOTH)
        right = ttk.Frame(self, padding=8)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        # ----- Shared main canvas (used by both tabs for live view) ---------
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

        # Status bar (shared)
        self.status_var = tk.StringVar(value="Ready. Pick an image or model.")
        self.status = ttk.Label(left, textvariable=self.status_var,
                                anchor=tk.W, relief=tk.SUNKEN)
        self.status.pack(fill=tk.X, pady=(6, 0))

        # ----- Tabbed controls (single-target vs staged curriculum) ---------
        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_single = ttk.Frame(self.notebook, padding=8)
        self.tab_staged = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.tab_single, text="Single image")
        self.notebook.add(self.tab_staged, text="Staged curriculum")

        # Switch the live canvas source + erase-routing when tab changes.
        self.notebook.bind("<<NotebookTabChanged>>",
                           lambda _e: self._on_tab_changed())

        # Track which tab owns the canvas so erase/double-click go to the
        # right state tensor.
        self.active_tab = "single"

        self._build_single_tab(self.tab_single)
        self._build_staged_tab(self.tab_staged)

        ttk.Button(right, text="Quit", command=self.destroy).pack(
            fill=tk.X, pady=(12, 0))

    # --------------------------------------------------- single tab -- #
    def _build_single_tab(self, parent):
        # Target image selector
        img_frame = ttk.LabelFrame(parent, text="Target image", padding=8)
        img_frame.pack(fill=tk.X)
        self.img_label = ttk.Label(img_frame, text="no image",
                                   anchor=tk.CENTER, width=22)
        self.img_label.pack()
        self.img_thumb = None  # keep ref
        ttk.Button(img_frame, text="Open image...",
                   command=self._on_open_image).pack(fill=tk.X, pady=(6, 0))

        # Training
        tr = ttk.LabelFrame(parent, text="Train", padding=8)
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
        md = ttk.LabelFrame(parent, text="Saved models", padding=8)
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
        pb = ttk.LabelFrame(parent, text="Playback", padding=8)
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

    # -------------------------------------------------- staged tab -- #
    def _build_staged_tab(self, parent):
        # --- sprite sheet selector -------------------------------------
        sheet = ttk.LabelFrame(parent, text="Sprite sheet (all.png)", padding=8)
        sheet.pack(fill=tk.X)
        sheet_row = ttk.Frame(sheet); sheet_row.pack(fill=tk.X)
        self.stage_sheet_var = tk.StringVar(
            value="sourceimg/all.png" if os.path.exists("sourceimg/all.png")
            else "select sprite sheet...")
        ttk.Entry(sheet_row, textvariable=self.stage_sheet_var,
                  width=18).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(sheet_row, text="...",
                   command=self._on_open_stage_sheet).pack(side=tk.LEFT, padx=(2, 0))

        # --- live preview thumbnail of current stage target -----------
        tgt = ttk.LabelFrame(parent, text="Current stage target", padding=8)
        tgt.pack(fill=tk.X, pady=8)
        self.stage_target_label = ttk.Label(tgt, text="(load sheet first)",
                                            anchor=tk.CENTER)
        self.stage_target_label.pack()
        self.stage_target_img = None  # tk photo ref

        # --- curriculum hyperparameters -------------------------------
        hp = ttk.LabelFrame(parent, text="Curriculum", padding=8)
        hp.pack(fill=tk.X)
        ttk.Label(hp, text="Stages:").grid(row=0, column=0, sticky=tk.W)
        self.st_n_var = tk.IntVar(value=11)
        ttk.Spinbox(hp, from_=2, to=64, width=4,
                    textvariable=self.st_n_var).grid(row=0, column=1, padx=4)
        ttk.Label(hp, text="Ep/stage:").grid(row=0, column=2, sticky=tk.W)
        self.st_eps_var = tk.IntVar(value=2000)
        ttk.Spinbox(hp, from_=50, to=20000, increment=200, width=6,
                    textvariable=self.st_eps_var).grid(row=0, column=3)
        ttk.Label(hp, text="Threshold:").grid(row=1, column=0, sticky=tk.W)
        self.st_thr_var = tk.DoubleVar(value=5e-3)
        ttk.Entry(hp, textvariable=self.st_thr_var, width=6).grid(
            row=1, column=1, padx=4)
        ttk.Label(hp, text="Step×:").grid(row=1, column=2, sticky=tk.W)
        self.st_fac_var = tk.DoubleVar(value=1.5)
        ttk.Spinbox(hp, from_=1.0, to=5.0, increment=0.1, width=4,
                    textvariable=self.st_fac_var).grid(row=1, column=3)

        self.stage_train_btn = ttk.Button(
            hp, text="Start staged training",
            command=self._on_start_staged_training)
        self.stage_train_btn.grid(row=2, column=0, columnspan=4,
                                  sticky=tk.EW, pady=(6, 0))
        self.stage_progress = ttk.Progressbar(hp, mode="determinate")
        self.stage_progress.grid(row=3, column=0, columnspan=4,
                                 sticky=tk.EW, pady=(6, 0))
        self.stage_status_var = tk.StringVar(value="staged: idle")
        ttk.Label(hp, textvariable=self.stage_status_var,
                  anchor=tk.W).grid(row=4, column=0, columnspan=4,
                                    sticky=tk.W, pady=(2, 0))

        # --- staged playback (use the live canvas like the single tab) -
        pb = ttk.LabelFrame(parent, text="Playback (staged)", padding=8)
        pb.pack(fill=tk.X, pady=8)
        row = ttk.Frame(pb); row.pack(fill=tk.X)
        self.staged_play_btn = ttk.Button(
            row, text="Play", command=self._on_staged_play, width=6)
        self.staged_play_btn.pack(side=tk.LEFT)
        ttk.Button(row, text="Step", command=self._on_staged_step).pack(side=tk.LEFT)
        ttk.Button(row, text="Reset", command=self._on_staged_reset).pack(side=tk.LEFT)

        spd = ttk.Frame(pb); spd.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(spd, text="Steps/frame:").pack(side=tk.LEFT)
        self.st_spf_var = tk.IntVar(value=2)
        ttk.Spinbox(spd, from_=1, to=10, width=4,
                    textvariable=self.st_spf_var).pack(side=tk.LEFT, padx=4)

        nxt = ttk.Frame(pb); nxt.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(nxt, text="Current stage:").pack(side=tk.LEFT)
        self.staged_stage_var = tk.IntVar(value=0)
        self.staged_stage_lbl = ttk.Label(nxt, textvariable=self.staged_stage_var,
                                          width=3)
        self.staged_stage_lbl.pack(side=tk.LEFT, padx=4)
        ttk.Button(nxt, text="Prev stage",
                   command=lambda: self._staged_change_stage(-1)).pack(side=tk.LEFT)
        ttk.Button(nxt, text="Next stage",
                   command=lambda: self._staged_change_stage(+1)).pack(side=tk.LEFT,
                                                                       padx=(2, 0))

        # --- model loader (reuse shared dropdown) ---------------------
        md = ttk.LabelFrame(parent, text="Saved staged models", padding=8)
        md.pack(fill=tk.X)
        self.staged_model_var = tk.StringVar()
        self.staged_model_box = ttk.Combobox(md, textvariable=self.staged_model_var,
                                             state="readonly", width=22)
        self.staged_model_box.pack(fill=tk.X)
        ttk.Button(md, text="Load selected staged model",
                   command=self._on_load_staged_model).pack(fill=tk.X, pady=(6, 0))
        ttk.Button(md, text="Refresh list",
                   command=self._refresh_staged_models_dropdown).pack(fill=tk.X)

    # -------------------------------------------------- tab routing -- #
    def _on_tab_changed(self):
        try:
            idx = self.notebook.index(self.notebook.select())
        except tk.TclError:
            return
        self.active_tab = "staged" if idx == 1 else "single"
        # When switching into the staged tab, show its state on the canvas.
        if self.active_tab == "staged":
            if self.staged_state is None:
                self.staged_state = make_seed(
                    GRID_SIZE, CHANNEL_N, device=self.device)
            self._redraw_staged()
        else:
            # Leave the single-tab state as it is on the canvas.
            if self.state is None and self.model is not None:
                self._on_reset()
        self._set_status(f"switched to '{self.active_tab}' tab")

    # ---------------------------------------------------- helpers ------- #
    def _refresh_models_dropdown(self):
        files = []
        if os.path.isdir("models"):
            files = sorted(f for f in os.listdir("models")
                           if f.endswith(".pth"))
        self.model_box["values"] = files
        if files and not self.model_var.get():
            self.model_var.set(files[0])
        # Also refresh the staged-tab dropdown (same folder for now).
        staged = [f for f in files if f.startswith("staged")]
        self.staged_model_box["values"] = staged
        if staged and not self.staged_model_var.get():
            self.staged_model_var.set(staged[0])

    def _refresh_staged_models_dropdown(self):
        # Same source as the single tab; mirror selection.
        self._refresh_models_dropdown()

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

    # ----------------------------------------------------- staged ---- #
    def _on_open_stage_sheet(self):
        path = filedialog.askopenfilename(
            title="Select sprite sheet (horizontal stack)",
            filetypes=[("Image files",
                       "*.png *.jpg *.jpeg *.webp *.bmp *.gif"),
                       ("All files", "*.*")])
        if path:
            self.stage_sheet_var.set(path)
            self._load_stage_sheet(path)

    def _load_stage_sheet(self, path: str) -> bool:
        n = int(self.st_n_var.get())
        try:
            self.stage_targets = load_stage_targets(path, n_stages=n)
            self.stage_sheet_path = path
        except Exception as exc:
            messagebox.showerror("Sprite sheet error", str(exc))
            self.stage_targets = None
            return False
        self._show_stage_target(0)
        self._set_stage_status(
            f"loaded {n} frames from {os.path.basename(path)}")
        return True

    def _show_stage_target(self, stage: int):
        """Render the target thumbnail for ``stage`` in the staged-tab frame."""
        if self.stage_targets is None:
            return
        from image_utils import to_rgb_display
        arr = to_rgb_display(self.stage_targets[stage:stage + 1])
        thumb = Image.fromarray(arr, mode="RGB").resize(
            (CANVAS_SIZE // 2, CANVAS_SIZE // 2), Image.NEAREST)
        self.stage_target_img = ImageTk.PhotoImage(thumb)
        self.stage_target_label.config(image=self.stage_target_img, text="")

    def _set_stage_status(self, msg: str):
        self.stage_status_var.set("staged: " + msg)

    def _draw_stage_preview(self, rgb: np.ndarray):
        """Push an HxWx3 uint8 preview array onto the shared main canvas."""
        img = Image.fromarray(rgb, mode="RGB").resize(
            (GRID_SIZE * CANVAS_SCALE, GRID_SIZE * CANVAS_SCALE),
            Image.NEAREST)
        self._tk_image = ImageTk.PhotoImage(img)
        self.canvas.itemconfig(self.canvas_image, image=self._tk_image)

    def _on_start_staged_training(self):
        if self.train_thread and self.train_thread.is_alive():
            messagebox.showwarning("Busy",
                                   "Another training is already running.")
            return
        sheet = self.stage_sheet_var.get()
        if not sheet or not os.path.exists(sheet):
            messagebox.showwarning("No sprite sheet",
                                   "Pick a horizontal sprite sheet first "
                                   "(e.g. sourceimg/all.png).")
            return

        n_stages = int(self.st_n_var.get())
        if self.stage_targets is None or self.stage_targets.shape[0] != n_stages:
            if not self._load_stage_sheet(sheet):
                return

        epochs_per_stage = int(self.st_eps_var.get())
        threshold = float(self.st_thr_var.get())
        step_factor = float(self.st_fac_var.get())

        # total = n_stages * epochs_per_stage, the progress bar uses this.
        total = n_stages * epochs_per_stage
        self.stage_progress["maximum"] = total
        self.stage_progress["value"] = 0
        self.stage_train_btn.config(state=tk.DISABLED,
                                    text="Training (live preview)...")
        self._set_stage_status(f"starting curriculum on {self.device}...")

        # Capture a CPU copy of the targets so the worker can train on CPU.
        targets = self.stage_targets.detach().cpu()

        # Files for the trained model + JSON schedule.
        ts = time.strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join("models", f"staged_{ts}.pth")
        sched_path = os.path.join(
            "models", f"staged_{ts}_stage_schedule.json")
        os.makedirs("models", exist_ok=True)

        def prog(stage, stage_epoch, total_epoch, loss):
            self.after(0, lambda s=stage, se=stage_epoch, te=total_epoch,
                       l=loss: self._on_staged_progress(s, se, te, l))

        def snap(stage, total_epoch, rgb):
            self.after(0, lambda s=stage, te=total_epoch, r=rgb:
                       self._on_staged_snapshot(s, te, r))

        def worker():
            from stage_train import (DEFAULT_STAGE_MAX_EPOCHS, train_staged)
            try:
                ca, sched = train_staged(
                    targets,
                    n_stages=n_stages,
                    epochs_per_stage=epochs_per_stage,
                    threshold=threshold,
                    step_factor=step_factor,
                    snapshot_dir=".temp",
                    snapshot_every=200,
                    save_path=save_path,
                    schedule_path=sched_path,
                    on_progress=prog,
                    on_snapshot=snap,
                    snapshot_preview_every=25,
                    snapshot_preview_steps=80,
                    force_cpu=True,
                )
                self.after(0, lambda: self._on_staged_done(ca, sched,
                                                           save_path, sched_path))
            except Exception as exc:
                self.after(0, lambda e=exc: self._on_staged_error(e))

        self.train_thread = threading.Thread(target=worker, daemon=True)
        self.train_thread.start()

    def _on_staged_progress(self, stage, stage_epoch, total_epoch, loss):
        self.stage_progress["value"] = total_epoch
        self._set_stage_status(
            f"stage {stage}  ep {stage_epoch}  total {total_epoch}  "
            f"loss={loss:.5f}")

    def _on_staged_snapshot(self, stage, total_epoch, rgb):
        self._show_stage_target(stage)
        self._draw_stage_preview(rgb)
        # also surface it in the main status bar
        self._set_status(f"[staged] stage {stage} ep {total_epoch}")

    def _on_staged_done(self, ca, sched, save_path, sched_path):
        self.stage_model = ca
        self._staged_schedule = sched           # cache for default step counts
        self.stage_train_btn.config(state=tk.NORMAL, text="Start staged training")
        self._refresh_models_dropdown()
        self.staged_model_var.set(os.path.basename(save_path))
        self.model_var.set(os.path.basename(save_path))
        ni = [s.converged_epochs for s in sched.stages]
        self._set_stage_status(
            f"done. N per stage = {ni}. Schedule saved.")
        messagebox.showinfo(
            "Staged training complete",
            f"Model: {save_path}\nSchedule: {sched_path}\n\n"
            f"Convergence epochs per stage:\n{ni}\n\n"
            "Inference step counts (1.5x): "
            f"{[sched.inference_steps(i) for i in range(sched.n_stage)]}")

    def _on_staged_error(self, exc):
        self.stage_train_btn.config(state=tk.NORMAL,
                                    text="Start staged training")
        self._set_stage_status(f"error: {exc}")
        messagebox.showerror("Staged training error", str(exc))

    def _on_load_staged_model(self):
        name = self.staged_model_var.get()
        if not name:
            messagebox.showwarning("No model", "No staged model selected.")
            return
        path = os.path.join("models", name)
        try:
            from stage_train import load_staged_model
            self.stage_model = load_staged_model(path, device=self.device)
            self._staged_schedule = None
        except Exception as exc:
            messagebox.showerror("Staged model error", str(exc))
            return
        self._on_staged_reset()
        # Switch the user into the staged tab so they can play.
        self.notebook.select(self.tab_staged)
        self._set_status(f"Loaded staged model: {name}")

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

    # ---------------------------------- staged playback --------- #
    def _on_staged_play(self):
        if self.stage_model is None:
            messagebox.showwarning("No model",
                                   "Train or load a staged model first.")
            return
        self.staged_playing = not self.staged_playing
        self.staged_play_btn.config(
            text="Pause" if self.staged_playing else "Play")

    def _on_staged_step(self):
        if self.stage_model is None:
            messagebox.showwarning("No model",
                                   "Train or load a staged model first.")
            return
        self._advance_staged()

    def _on_staged_reset(self):
        if self.stage_model is None:
            return
        self.staged_state = make_seed(GRID_SIZE, CHANNEL_N, device=self.device)
        self.staged_stage_var.set(0)
        self.staged_playing = False
        self.staged_play_btn.config(text="Play")
        self._redraw_staged()

    def _staged_change_stage(self, delta: int):
        if self.stage_model is None:
            return
        cur = int(self.staged_stage_var.get())
        n = self.stage_model.stage_n
        new = max(0, min(n - 1, cur + delta))
        self.staged_stage_var.set(new)
        self._show_stage_target(new)
        self._set_status(f"staged stage -> {new}/{n - 1}")

    def _advance_staged(self):
        if self.staged_state is None or self.stage_model is None:
            return
        # Keep model & state on the same device as the GUI's preferred device.
        self.stage_model = self.stage_model.to(self.device)
        self.staged_state = self.staged_state.to(self.device)
        stage = int(self.staged_stage_var.get())
        with torch.no_grad():
            for _ in range(int(self.st_spf_var.get())):
                self.staged_state = self.stage_model(self.staged_state,
                                                     stage=stage)
        self._redraw_staged()

    def _redraw_staged(self):
        if self.staged_state is None:
            return
        img = state_to_pil(self.staged_state, scale=CANVAS_SCALE)
        self._tk_image = ImageTk.PhotoImage(img)
        self.canvas.itemconfig(self.canvas_image, image=self._tk_image)

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
        # Route to whichever tab currently owns the canvas.
        if self.active_tab == "staged":
            state = self.staged_state
        else:
            state = self.state
        if state is None:
            return
        gx, gy = self._canvas_to_grid(event)
        radius = 5
        lo_x = max(0, gx - radius); hi_x = min(GRID_SIZE, gx + radius + 1)
        lo_y = max(0, gy - radius); hi_y = min(GRID_SIZE, gy + radius + 1)
        with torch.no_grad():
            state[:, :, lo_y:hi_y, lo_x:hi_x] = 0.0
        if self.active_tab == "staged":
            self._redraw_staged()
        else:
            self._redraw()

    def _on_double_click(self, event):
        gx, gy = self._canvas_to_grid(event)
        with torch.no_grad():
            if self.active_tab == "staged":
                self.staged_state = make_seed(
                    GRID_SIZE, CHANNEL_N, device=self.device)
                self.staged_state[:, :, gy, gx] = 0.0
                self.staged_state[:, 3:, gy, gx] = 1.0
                self._redraw_staged()
            else:
                self.state = make_seed(
                    GRID_SIZE, CHANNEL_N, device=self.device)
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
        if self.active_tab == "staged":
            if self.staged_playing and self.stage_model is not None:
                try:
                    self._advance_staged()
                except Exception as exc:
                    self._set_status(f"error: {exc}")
                    self.staged_playing = False
                    self.staged_play_btn.config(text="Play")
        else:
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
