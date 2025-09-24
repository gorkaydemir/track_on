"""
Track-On2 demo script

Example:
    python demo.py \
        --video davis_videos/motocross-bumps.mp4 \
        --config ./config/test.yaml \
        --ckpt path/to/checkpoint.pth \
        --output debug_output.mp4 \
        --use-grid false \
        --point-size 100
"""

import argparse
import os
import sys
from typing import Optional

import numpy as np
import torch
from torchvision.io import read_video

from model.trackon_predictor import Predictor
from utils.train_utils import load_args_from_yaml
from utils.vis_utils import plot_tracks_wo_tail, save_video


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track-On2 demo")
    p.add_argument("--video", required=True, type=str, help="Path to input video (e.g., .mp4)")
    p.add_argument("--config", default="./config/test.yaml", type=str, help="Path to model config .yaml")
    p.add_argument("--ckpt", required=True, type=str, help="Path to Track-On2 checkpoint .pth")
    p.add_argument("--output", default="demp_output.mp4", type=str, help="Path to output visualization video")
    p.add_argument("--use-grid", default=False, type=lambda x: str(x).lower() in {"1","true","yes","y"},
                   help="If true, uses a uniform grid of queries (set queries=None).")
    p.add_argument("--point-size", default=100, type=int, help="Dot size for visualization.")
    return p.parse_args()


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_video_thwc(path: str) -> torch.Tensor:
    """Load video as (T, H, W, C) and ensure 3 channels."""
    video, _, _ = read_video(path, output_format="THWC")
    return video[..., :3]  # (T, H, W, 3), uint8


def ensure_paths(args: argparse.Namespace) -> None:
    if not os.path.isfile(args.video):
        sys.exit(f"[ERROR] Video not found: {args.video}")
    if not os.path.isfile(args.config):
        sys.exit(f"[ERROR] Config not found: {args.config}")
    if not os.path.isfile(args.ckpt):
        sys.exit(f"[ERROR] Checkpoint not found: {args.ckpt}")
    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)


def build_manual_queries(device: str) -> torch.Tensor:
    """
    Example manual queries in (1, N, 3) with (t, x, y).
    Adjust coordinates to your video if needed.
    """
    q = torch.tensor(
        [
            [0, 190, 190],
            [0, 190, 190],
            [0, 200, 190],
            [0, 190, 200],
            [0, 200, 200],
            [0, 190, 210],
            [0, 200, 210],
        ],
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)  # (1, N, 3)
    return q


def main() -> None:
    args = parse_args()
    ensure_paths(args)

    device = pick_device()
    print(f"[Info] Using device: {device}")

    # === Read Video ===
    print(f"[Info] Loading video: {args.video}")
    video_thwc = load_video_thwc(args.video)  # (T, H, W, 3), uint8
    vis_rgb_thwc = video_thwc.numpy()         # keep THWC copy for visualization

    # Convert for model: (T, 3, H, W), float32
    video_tchw = video_thwc.permute(0, 3, 1, 2).contiguous().float()

    # === Initialize the model ===
    print(f"[Info] Loading model config: {args.config}")
    model_args = load_args_from_yaml(args.config)

    print(f"[Info] Initializing model with checkpoint: {args.ckpt}")
    model = Predictor(model_args, checkpoint_path=args.ckpt).to(device).eval()

    # === Initialize the Queries ===
    if args.use_grid:
        queries: Optional[torch.Tensor] = None  # uniform grid via model's default
        print("[Info] Using uniform grid queries (queries=None).")
    else:
        queries = build_manual_queries(device)
        print(f"[Info] Using manual queries with shape: {tuple(queries.shape)}")

    # === Run the model ===
    print("[Info] Running inference...")
    with torch.no_grad():
        vid_batch = video_tchw.unsqueeze(0).to(device)  # (1, T, 3, H, W)
        tracks, visibles = model(vid_batch, queries=queries)  # (1, T, N, 2), (1, T, N)

    # === Visualize the predictions ===
    print("[Info] Visualizing tracks...")
    tracks_nt2 = tracks[0].detach().cpu().numpy().transpose(1, 0, 2)       # (N, T, 2)
    occluded_nt = (1 - visibles[0].detach().cpu().numpy()).transpose(1, 0) # (N, T)

    video_track = plot_tracks_wo_tail(
        vis_rgb_thwc,      # (T, H, W, 3) uint8
        tracks_nt2,        # (N, T, 2)
        occluded_nt,       # (N, T)
        point_size=args.point_size
    )

    print(f"[Info] Saving result to: {args.output}")
    # No FPS argument passed:
    save_video(video_track, args.output)

    print("[Done] Demo finished successfully.")


if __name__ == "__main__":
    main()