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
                         make_seed, state_to_pil)
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
    args = parser.parse_args()

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
