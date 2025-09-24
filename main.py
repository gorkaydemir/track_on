import os
import sys
import numpy as np
from tqdm import tqdm
import time
import datetime
import math

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn

import wandb

from pathlib import Path

import argparse

from utils.train_utils import init_distributed_mode, fix_random_seeds
from utils.train_utils import get_dataloaders, get_scheduler
from utils.train_utils import restart_from_checkpoint, save_on_master
from utils.train_utils import load_args_from_yaml

from utils.log_utils import init_wandb, Loss_Tracker

from utils.coord_utils import get_queries, get_points_on_a_grid

from utils.eval_utils import Evaluator, compute_tapvid_metrics

from model.trackon import Track_On2
from model.loss import Loss_Function


def train(args, train_dataloader, model, optimizer, lr_scheduler, scaler, loss_tracker):
    model.train()

    loss_fn = Loss_Function(args)
    train_dataloader = tqdm(train_dataloader, disable=args.rank != 0, file=sys.stdout)

    for i, (video, tracks, visibility, k_points) in enumerate(train_dataloader):
        video = video.cuda(non_blocking=True)               # (B, T, 3, H, W)
        tracks = tracks.cuda(non_blocking=True)             # (B, T, N, 2)
        visibility = visibility.cuda(non_blocking=True)     # (B, T, N)
        k_points = k_points.cuda(non_blocking=True)         # (B,)

        min_k = k_points.min()
        tracks = tracks[:, :, :min_k]
        visibility = visibility[:, :, :min_k]

        B, T, N, _ = tracks.shape

        # === Scaling ===
        H_in, W_in = video.shape[-2], video.shape[-1]
        H, W = args.input_size[0], args.input_size[1]

        tracks_gt = tracks.clone()
        tracks_gt[..., 0] = tracks_gt[..., 0] * (W / W_in)
        tracks_gt[..., 1] = tracks_gt[..., 1] * (H / H_in)
        # === === ===

        queries = get_queries(tracks, visibility)             # (B, N, 3)
        query_times = queries[:, :, 0].long()        # (B, N)

        # === Forward pass ===
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
            out = model(video, queries, p_mask=0.1)              # (B, N, P), (B, N)
            L_p, L_p2, L_vis, L_o, L_u, L_topk_u, L_topk_rank = loss_fn(out, tracks_gt, 
                                                                        visibility, query_times)
            
        loss = L_p + L_p2 + L_vis + L_o + L_u + L_topk_u + L_topk_rank
      
        if args.amp:
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        # ======== ======== ======

        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        loss_tracker.update(L_p, L_p2, L_vis, L_o, L_u, L_topk_u, L_topk_rank)
        loss_tracker.log()

        train_dataloader.set_description(f"Loss: {loss.item():.5f}")

        

@torch.no_grad()
def evaluate(args, val_dataloader, model, epoch, verbose=False):
    model.eval()

    evaluator = Evaluator()
    total_frames = 0
    total_time = 0

    for j, (video, trajectory, visibility, query_points_i) in enumerate(tqdm(val_dataloader, disable=verbose, file=sys.stdout)):
        # Timer start
        start_time = time.time()
        total_frames += video.shape[1]

        query_points_i = query_points_i.cuda(non_blocking=True)      # (1, N, 3)
        trajectory = trajectory.cuda(non_blocking=True)              # (1, T, N, 2)
        visibility = visibility.cuda(non_blocking=True)              # (1, T, N)
        video = video.cuda(non_blocking=True)                    # (1, T, 3, H, W)
        B, T, N, _ = trajectory.shape
        _, _, _, H, W = video.shape
        device = video.device

        # Change (t, y, x) to (t, x, y)
        queries = query_points_i.clone().float()
        queries = torch.stack([queries[:, :, 0], queries[:, :, 2], queries[:, :, 1]], dim=2).to(device)       # (1, N, 3)

        # Support grid
        K = 20
        extra_queries = get_points_on_a_grid(K, (H, W), device)             # (1, K ** 2, 2)
        extra_queries = torch.cat([torch.zeros(1, int(K ** 2), 1, device=device), extra_queries], dim=-1).to(device)
        queries = torch.cat([queries, extra_queries], dim=1)

        # out, _ = model.module(video, queries, p_mask=0)
        out = model.module.forward_online(video, queries)

        pred_trajectory = out["P"][:, :, :N]                                           # (1, T, N, P)
        pred_visibility = out["V_logit"][:, :, :N].sigmoid() >= args.delta_v       # (1, T, N)

        # Timer end
        total_time += time.time() - start_time

        # === === ===
        # From CoTracker
        traj = trajectory.clone()
        query_points = query_points_i.clone().cpu().numpy()
        gt_tracks = traj.permute(0, 2, 1, 3).cpu().numpy()
        gt_occluded = torch.logical_not(visibility.clone().permute(0, 2, 1)).cpu().numpy()
        pred_occluded = torch.logical_not(pred_visibility.clone().permute(0, 2, 1)).cpu().numpy()
        pred_tracks = pred_trajectory.permute(0, 2, 1, 3).cpu().numpy()
        # === === ===

        out_metrics = compute_tapvid_metrics(query_points, gt_occluded, gt_tracks, pred_occluded, pred_tracks, "first")
        if verbose:
            print(f"Video {j}/{len(val_dataloader)}: AJ: {out_metrics['average_jaccard'][0] * 100:.2f}, delta_avg: {out_metrics['average_pts_within_thresh'][0] * 100:.2f}, OA: {out_metrics['occlusion_accuracy'][0] * 100:.2f}", flush=True)
        evaluator.update(out_metrics)
        
    fps = total_frames / total_time
    print(f"Evaluation FPS: {fps:.2f}", flush=True)

    results = evaluator.get_results(epoch)
    smaller_delta_avg = results["delta_avg"]
    aj = results["aj"]
    oa = results["oa"]

    return smaller_delta_avg, aj, oa


def main_worker(args):
    init_distributed_mode(args)
    fix_random_seeds(args.seed)

    start_time = time.time()

    # === Data ===
    train_dataloader, val_dataloader = get_dataloaders(args)
    if train_dataloader is not None:
        print(f"Total number of iterations: {len(train_dataloader) * args.epoch_num / 1000:.1f}K")
    # === === ===
    
    # === Model & Training ===
    model = Track_On2(args).to(args.gpu)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = get_scheduler(args, optimizer, train_dataloader, constant=False)
    scaler = torch.GradScaler() if args.amp else None
    loss_tracker = Loss_Tracker(args.rank)
    init_wandb(args)

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 10**6:.2f}M")
    print(f"Number of learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 10**6:.2f}M")
    print()
    # === === ===

    # === Load from checkpoint ===
    to_restore = {"epoch": 0, "iteration": 0, "best_aj": [-1, -1], "best_oa": [-1, -1], "best_delta_avg": [-1, -1]}

    if args.checkpoint_path is not None:
        if not os.path.exists(args.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint_path}")
        
        restart_from_checkpoint(remove_module=False,
                                checkpoint_path=args.checkpoint_path, 
                                run_variables=to_restore, 
                                model=model,
                                scaler=scaler,
                                optimizer=optimizer, 
                                scheduler=lr_scheduler)
    # === === ===


    # === Start training ===
    start_epoch = to_restore["epoch"]
    best_models = {
        "aj": to_restore["best_aj"],
        "oa": to_restore["best_oa"],
        "delta_avg": to_restore["best_delta_avg"]
    }
    loss_tracker.iteration = to_restore["iteration"]


    print("Training starts")
    for epoch in range(start_epoch, args.epoch_num):
        train_dataloader.sampler.set_epoch(epoch)

        print(f"=== === Epoch {epoch} === ===")

        # === === Training === ===

        train(args, train_dataloader, model, optimizer, lr_scheduler, scaler, loss_tracker)
        print()
        # === === ===
        
        # === Evaluation ===
        if args.rank == 0:

            smaller_delta_avg, aj, oa = evaluate(args, val_dataloader, model, epoch)

            # === Update Best Models ===
            best_aj = False
            best_oa = False
            best_delta_avg = False
            if aj > best_models["aj"][1]:
                best_models["aj"] = [epoch, aj]
                best_aj = True

            if oa > best_models["oa"][1]:
                best_models["oa"] = [epoch, oa]
                best_oa = True

            if smaller_delta_avg > best_models["delta_avg"][1]:
                best_models["delta_avg"] = [epoch, smaller_delta_avg]
                best_delta_avg = True
            # === === ===
            
            # # === Save Model ===
            save_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": lr_scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "epoch": epoch + 1,
                "best_aj": best_models["aj"],
                "best_oa": best_models["oa"],
                "best_delta_avg": best_models["delta_avg"],
                "iteration": loss_tracker.iteration,
            }
        
            save_on_master(save_dict, os.path.join(args.model_save_path, "checkpoint.pt"))

            # save if scaores are best
            if best_aj: save_on_master(save_dict, os.path.join(args.model_save_path, "best_aj.pt"))
            if best_oa: save_on_master(save_dict, os.path.join(args.model_save_path, "best_oa.pt"))
            if best_delta_avg: save_on_master(save_dict, os.path.join(args.model_save_path, "best_delta.pt"))

            # # === === ===

        dist.barrier()

        print(f"=== === === === === ===")
        print()

    # print best results
    if args.rank == 0:
        print("Best Results")
        print(f"Best AJ: {best_models['aj'][1]:.3f} at epoch {best_models['aj'][0]}")
        print(f"Best OA: {best_models['oa'][1]:.3f} at epoch {best_models['oa'][0]}")
        print(f"Best Smaller Delta Avg: {best_models['delta_avg'][1]:.3f} at epoch {best_models['delta_avg'][0]}")
    
    wandb.finish()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")
    dist.destroy_process_group()


if __name__ == '__main__':
    # args = get_args()

    parser = argparse.ArgumentParser(description="Training for Track-On2")

    parser.add_argument("--config_path", type=str, required=True, help='Path to the config file')
    parser.add_argument("--movi_f_root", type=str, help='Path to the MOVi-F dataset')
    parser.add_argument("--tapvid_root", type=str, help='Path to the TAP-Vid DAVIS dataset')
    parser.add_argument("--model_save_path", type=str, help='Path to save the model checkpoints')
    parser.add_argument("--checkpoint_path", type=str, default=None, help='Path to the model checkpoint to resume training')
    args_cmd = parser.parse_args()

    args = load_args_from_yaml(args_cmd.config_path)
    
    args.movi_f_root = args_cmd.movi_f_root
    args.tapvid_root = args_cmd.tapvid_root
    args.model_save_path = args_cmd.model_save_path
    args.checkpoint_path = args_cmd.checkpoint_path

    args.gpus = torch.cuda.device_count()
    Path(args.model_save_path).mkdir(parents=True, exist_ok=True)

    main_worker(args)