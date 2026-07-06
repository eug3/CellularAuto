"""Entry point for the Growing Neural Cellular Automata project.

Usage
-----
    # 1. Just open the GUI (you pick the image there):
    python main.py

    # 2. Open the GUI with a preset image:
    python main.py --image targets/heart.png

    # 3. Headless: train a model on a given image, no GUI:
    python main.py --image targets/heart.png --train --epochs 4000

    # 4. Headless: export a handful of growth-frame PNGs from a trained model:
    python main.py --image targets/heart.png --play --model models/heart_xxx.pth
"""
from __future__ import annotations

import argparse
import os
import sys

# Make sibling modules importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from image_utils import (CHANNEL_N, GRID_SIZE, get_device, load_target,
                         load_stage_targets, make_seed, state_to_pil)
from train import DEFAULT_EPOCHS, load_model, save_model, train


def cmd_train(args):
    device = get_device(args.device)
    target = load_target(args.image, device=device)
    save_path = args.save
    if save_path is None:
        base = os.path.splitext(os.path.basename(args.image))[0]
        save_path = os.path.join("models", f"{base}.pth")
    train(
        target,
        epochs=args.epochs,
        device=device,
        regenerate=args.regen,
        snapshot_dir=".temp" if not args.no_snapshot else None,
        save_path=save_path,
    )
    print("Done. Model at", save_path)


def cmd_play(args):
    device = get_device(args.device)
    target = load_target(args.image, device=device)
    if args.model is None:
        raise SystemExit("--play requires --model <path>.")
    from ca_model import CAModel
    ca = load_model(args.model, device=device)
    ca.eval()
    state = make_seed(GRID_SIZE, CHANNEL_N, device=device)
    os.makedirs(args.out, exist_ok=True)
    n = max(args.steps, 1)
    for i in range(n):
        with __import__("torch").no_grad():
            state = ca(state)
        img = state_to_pil(state, scale=6)
        img.save(os.path.join(args.out, f"frame_{i:04d}.png"))
    print(f"Wrote {n} frames to {args.out}/")


def cmd_staged_train(args):
    """Curriculum-train a StageCAModel over all 11 sprite frames."""
    from stage_train import (DEFAULT_SAVE_NAME, DEFAULT_STEP_FACTOR,
                             DEFAULT_THRESHOLD, N_STAGE, SCHEDULE_NAME,
                             train_staged)
    device = get_device(args.device)
    targets = load_stage_targets(args.image, n_stages=args.stages, device=device)

    save_path = args.save or os.path.join("models", f"{DEFAULT_SAVE_NAME}.pth")
    schedule_path = os.path.join(
        os.path.dirname(save_path) or ".",
        os.path.splitext(os.path.basename(save_path))[0] + "_" + SCHEDULE_NAME)

    train_staged(
        targets,
        n_stages=args.stages,
        epochs_per_stage=args.epochs_per_stage,
        threshold=args.threshold,
        step_factor=args.step_factor,
        device=device,
        regenerate=args.regen,
        snapshot_dir=".temp" if not args.no_snapshot else None,
        snapshot_every=args.snapshot_every,
        save_path=save_path,
        schedule_path=schedule_path,
        force_cpu=False,
    )
    print("Done. Staged model at", save_path)
    print("       Schedule at   ", schedule_path)


def cmd_staged_play(args):
    """Run the 11-stage inference pipeline using the trained model + schedule."""
    from stage_train import load_staged_model, run_pipeline
    device = get_device(args.device)
    ca = load_staged_model(args.model, device=device)

    # Per-stage step counts may come from CLI (comma list) or from JSON file.
    if args.steps_csv:
        steps_list = [int(s) for s in args.steps_csv.split(",")]
    elif args.schedule:
        import json
        with open(args.schedule) as f:
            sched = json.load(f)
        steps_list = sched.get("inference_steps")
        if steps_list is None:
            raise SystemExit("Schedule JSON missing 'inference_steps' field.")
    else:
        raise SystemExit("--staged-play requires either --steps-csv or --schedule.")

    frames = run_pipeline(ca, steps_list, device=device,
                          out_dir=args.out, frame_every=args.frame_every,
                          extra_steps_per_stage=args.extra_steps)
    print(f"Wrote {len(frames)} frames to {args.out}/")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", help="path to target image")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "mps", "cuda", "cpu"])
    sub = parser.add_argument_group("headless modes")
    sub.add_argument("--train", action="store_true",
                     help="train a model on --image without opening the GUI")
    sub.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    sub.add_argument("--regen", action="store_true",
                     help="train with random damage (Experiment 3 - regeneration)")
    sub.add_argument("--no-snapshot", action="store_true",
                     help="don't write progress snapshots to .temp/")
    sub.add_argument("--save", help="output model path (default models/<name>.pth)")
    sub.add_argument("--play", action="store_true",
                     help="run a trained model headless and dump frames")
    sub.add_argument("--model", help="model weights path for --play")
    sub.add_argument("--steps", type=int, default=96,
                     help="number of growth steps for --play")
    sub.add_argument("--out", default="out_frames",
                     help="output dir for --play frames")
    # ------------------------------------------------------------------ #
    # Staged (multi-morph) subcommands
    # ------------------------------------------------------------------ #
    staged = parser.add_argument_group("staged (curriculum) modes")
    staged.add_argument("--staged-train", action="store_true",
                        help="curriculum-train a StageCAModel on the sprite-sheet --image")
    staged.add_argument("--stages", type=int, default=11,
                        help="number of horizontal frames (default 11)")
    staged.add_argument("--epochs-per-stage", type=int, default=4000,
                        help="max epochs per stage before giving up")
    staged.add_argument("--threshold", type=float, default=1.6e-3,
                        help="loss threshold below which a stage is considered converged")
    staged.add_argument("--step-factor", type=float, default=1.5,
                        help="inference-step multiplier over converge epochs (default 1.5)")
    staged.add_argument("--snapshot-every", type=int, default=200,
                        help="progress snapshot cadence (epochs)")
    staged.add_argument("--staged-play", action="store_true",
                        help="run the staged inference pipeline")
    staged.add_argument("--schedule",
                        help="path to *_stage_schedule.json for --staged-play")
    staged.add_argument("--steps-csv",
                        help="comma-separated per-stage step counts (alt to --schedule)")
    staged.add_argument("--frame-every", type=int, default=4,
                        help="save one frame every N steps during --staged-play")
    staged.add_argument("--extra", type=int, default=0,
                        help="extra steady-state steps per stage at inference")
    args = parser.parse_args()

    if args.staged_train:
        if not args.image:
            parser.error("--staged-train requires --image (the sprite sheet)")
        cmd_staged_train(args)
        return
    if args.staged_play:
        if not args.model:
            parser.error("--staged-play requires --model")
        cmd_staged_play(args)
        return
    if args.train:
        if not args.image:
            parser.error("--train requires --image")
        cmd_train(args)
        return
    if args.play:
        if not args.image or not args.model:
            parser.error("--play requires --image and --model")
        cmd_play(args)
        return

    # Default: launch the GUI.
    from gui import CAApp
    app = CAApp(initial_image=args.image,
                default_epochs=args.epochs,
                prefer_device=args.device)
    app.mainloop()


if __name__ == "__main__":
    main()
