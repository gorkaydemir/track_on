import torch
from torch.utils import data
import time

import argparse
import yaml
from argparse import Namespace

from utils.coord_utils import get_points_on_a_grid
from utils.train_utils import load_args_from_yaml
from model.trackon_predictor import Predictor
from dataset.tapvid import TAPVid

def get_args():
    parser = argparse.ArgumentParser(description="Benchmark script for Track-On2")
    parser.add_argument('--config_path', type=str, required=True, default="./config/test.yaml", help='Path to the model config file')
    parser.add_argument('--model_checkpoint_path', type=str, required=True, help='Path to the model checkpoint file')
    parser.add_argument('--davis_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--N_sqrt', type=int, default=8, help='Number of queries to use for evaluation')
    parser.add_argument('--memory_size', type=int, default=24, help='Number of queries to use for evaluation')

    return parser.parse_args()


if __name__ == "__main__":
    eval_args = get_args()
    model_args = load_args_from_yaml(eval_args.config_path)
    model_args.M_i = eval_args.memory_size

    print("=== Benchmarking Track-On2 ===")
    print(f"Memory size: {model_args.M_i}, N: {eval_args.N_sqrt**2}")

    
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('high')
    model = Predictor(model_args, checkpoint_path=eval_args.model_checkpoint_path, support_grid_size=0)
    model = model.cuda()

    # ---- dataloader tweaks (optional but recommended) ----
    dataset = TAPVid(None, data_root=eval_args.davis_path, dataset_type="davis")
    dataloader = data.DataLoader(dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=8,
                                    pin_memory=False)

    # ---- metrics we’ll accumulate ----
    total_frames = 0
    total_time_model_ms = 0.0      # pure model forward time
    total_time_e2e_ms = 0.0        # includes H2D + postproc

    # reset CUDA peak memory tracker
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()  # clean slate

    # (Optional) warmup a couple of iterations to populate autotuners / compile graphs
    _warmup = 2
    _end_at = 15

    for i, (video, trajectory, visibility, query_points_i) in enumerate(dataloader):
        t0_e2e = time.perf_counter()

        # Host->Device (pinned + non_blocking=True overlaps well)
        trajectory = trajectory.cuda(non_blocking=True)              # (1, T, N, 2)
        video = video.cuda(non_blocking=True)                        # (1, T, 3, H, W)

        # Prepare queries
        B, T, N, _ = trajectory.shape
        _, _, _, H, W = video.shape
        device = video.device

        queries = get_points_on_a_grid(eval_args.N_sqrt, (H, W), device)             # (1, K**2, 2)
        queries = torch.cat([torch.zeros(1, int(eval_args.N_sqrt**2), 1, device=device), queries], dim=-1)

        # ---- model timing (GPU-accurate via CUDA events) ----
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)

        # Use inference_mode() for best CPU/GPU-side overhead reduction
        with torch.inference_mode():
            start.record()
            pred_trajectory, pred_visibility = model(video, queries)
            end.record()
            torch.cuda.synchronize()
            iter_model_ms = start.elapsed_time(end)  # ms

        # End-to-end wall time for this iteration (sec -> ms)
        iter_e2e_ms = (time.perf_counter() - t0_e2e) * 1000.0

        # Skip warmup when accumulating stats (avoids first-iter compile/bench costs)
        if i >= _warmup:
            total_frames += int(T)
            total_time_model_ms += iter_model_ms
            total_time_e2e_ms += iter_e2e_ms

        if i == _end_at:
            break

    # Final stats
    peak_mem_bytes = torch.cuda.max_memory_allocated()
    peak_mem_gb = peak_mem_bytes / (1024**3)

    model_fps = total_frames / (total_time_model_ms / 1000.0) if total_time_model_ms > 0 else float('nan')
    e2e_fps   = total_frames / (total_time_e2e_ms / 1000.0) if total_time_e2e_ms > 0 else float('nan')

    print(f"[Speed] Model-only FPS: {model_fps:.2f} frames/s")
    print(f"[Speed] End-to-end FPS: {e2e_fps:.2f} frames/s (includes H2D + postproc)")
    print(f"[Memory] Peak GPU memory: {peak_mem_gb:.2f} GB")