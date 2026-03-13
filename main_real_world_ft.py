import os
import sys
import numpy as np
from tqdm import tqdm
import time
import datetime
import math
import cv2

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
from utils.train_utils import load_args_from_yaml, load_pretrained_weights

from utils.log_utils import init_wandb, Loss_Tracker

from utils.coord_utils import get_points_on_a_grid

from utils.eval_utils import Evaluator, compute_tapvid_metrics

from model.trackon import Track_On2
from model.loss import Loss_Function

# Import ensemble models

from utils.train_utils import load_args_from_yaml
from verifier.verifier_predictor import VerifierPredictor
from model.trackon_predictor import Predictor as Trackon_Predictor
from ensemble.bootstapir.bootstapir_predictor import TAPIRPredictor
from ensemble.tapnext.tapnext_predictor import TAPNextPredictor
from ensemble.cotracker import CoTracker_Predictor
from ensemble.locotrack.locotrack_predictor import LocoTrackPredictor
from ensemble.alltracker.alltracker_predictor import AllTrackerPredictor
from ensemble.ensemble_predictor import Ensemble_Predictor



def prepare_models(args):
    models = {}

    # === Pre-trained Track-On2 ===
    trackon2_args = load_args_from_yaml(args.trackon2_config_path)
    trackon2_args.M_i = 24
    model = Trackon_Predictor(trackon2_args, checkpoint_path=args.trackon2_checkpoint_path, support_grid_size=10).cuda()
    models["trackon2"] = model
    print(f"Loaded Trackon2 model")

    # === BootsTAP-Next ===
    model = TAPNextPredictor(args.bootstapnext_checkpoint_path).cuda()
    models["bootstapnext"] = model
    print(f"Loaded TAP-Next model")

    # === BootsTAPIR ===
    model = TAPIRPredictor(args.bootstapir_checkpoint_path).cuda()
    models["bootstapir"] = model
    print(f"Loaded BootsTAPIR model")

    # === CoTracker3 (Window) ===
    model = CoTracker_Predictor(windowed=True).cuda()
    models["cotracker3_window"] = model
    print(f"Loaded CoTracker3 Windowed model")

    # === Anthro-LocoTrack ===
    model = LocoTrackPredictor(args.anthro_locotrack_checkpoint_path).cuda()
    models["locotrack"] = model
    print(f"Loaded LocoTrack model")

    # === AllTracker ===
    model = AllTrackerPredictor(checkpoint_path=args.alltracker_checkpoint_path).cuda()
    models["alltracker"] = model
    print(f"Loaded AllTracker model")
    
    # === Verifier ===
    verifier_args = load_args_from_yaml(args.verifier_config_path)
    verifier_model = VerifierPredictor(verifier_args, checkpoint_path=args.verifier_checkpoint_path)
    verifier_model = verifier_model.cuda()
    print(f"Loaded Verifier model")
    models["verifier"] = verifier_model

    return models

def fine_tune(args, train_dataloader, model, ensemble_predictor, optimizer, lr_scheduler, scaler, loss_tracker, epoch, total_iterations):
    model.train()

    loss_fn = Loss_Function(args)
    steps_per_epoch = len(train_dataloader)
    train_dataloader = tqdm(train_dataloader, disable=args.rank != 0, file=sys.stdout)

    for i, (video_aug, video_clean, queries, tracks_gt_batch, visibility_gt_batch, is_synthetic) in enumerate(train_dataloader):
        # === Loss weighting schedule ===
        # Only active when syn_real_training=True AND schedule_syn_weight=True.
        # syn_weight:  1.0 → 0.0  (decreases linearly)
        # real_weight: 1.0 → 2.0  (increases linearly)
        use_schedule = getattr(args, 'syn_real_training', False) and getattr(args, 'schedule_syn_weight', False)
        if use_schedule:
            current_step = epoch * steps_per_epoch + i
            progress = min(current_step / max(total_iterations - 1, 1), 1.0)
            syn_weight  = 1.0 - progress   # 1.0 → 0.0
            real_weight = 1.0 + progress   # 1.0 → 2.0
        else:
            syn_weight  = 1.0
            real_weight = 1.0
        # === === ===
        # video_aug:          (B, T, 3, H, W) - augmented video
        # video_clean:        (B, T, 3, H, W) - clean video (only cropped; unused for synthetic)
        # queries:            (B, N, 3)        - query points in (t, x, y) format
        # tracks_gt_batch:    (B, T, N, 2)     - GT trajectories (non-zero only for synthetic)
        # visibility_gt_batch:(B, T, N)        - GT visibility   (non-zero only for synthetic)
        # is_synthetic:       (B,) bool        - True when Movi-F sample

        video_aug   = video_aug.cuda(non_blocking=True)
        video_clean = video_clean.cuda(non_blocking=True)
        queries     = queries.cuda(non_blocking=True)

        B, T, N = queries.shape[0], video_aug.shape[1], queries.shape[1]
        H, W = video_aug.shape[-2:]

        # Homogeneous batch: all samples are the same type (real or synthetic).
        synthetic_batch = is_synthetic[0].item()

        # === Scale queries to model's input size ===
        H_model, W_model = args.input_size

        queries_scaled = queries.clone()
        queries_scaled[..., 1] *= (W_model / W)  # x coordinate
        queries_scaled[..., 2] *= (H_model / H)  # y coordinate

        query_times = queries_scaled[:, :, 0].long()  # (B, N)

        if synthetic_batch:
            # ── Synthetic (Movi-F) path ───────────────────────────────────────
            # Use ground-truth tracks and visibility directly; no ensemble needed.
            tracks_gt_batch    = tracks_gt_batch.cuda(non_blocking=True)    # (B, T, N, 2)
            visibility_gt_batch = visibility_gt_batch.cuda(non_blocking=True)  # (B, T, N)

            # Scale GT tracks to model input size (no-op when Movi-F crop == input_size)
            tracks_gt = tracks_gt_batch.clone()
            tracks_gt[..., 0] *= (W_model / W)
            tracks_gt[..., 1] *= (H_model / H)

            visibility_for_loss = visibility_gt_batch.bool()

            # Forward pass – augmented video already produced by Movi-F pipeline
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                out = model(video_aug, queries_scaled, p_mask=0.1)
                L_p, L_p2, L_vis, L_o, L_u, L_topk_u, L_topk_rank = loss_fn(
                    out, tracks_gt, visibility_for_loss, query_times)

            # All losses are supervised with ground truth; scale by synthetic weight
            loss = syn_weight * (L_p + L_p2 + L_vis + L_o + L_u + L_topk_u + L_topk_rank)

        else:
            # ── Real-world path ───────────────────────────────────────────────
            # Generate pseudo-ground-truth via ensemble + verifier on clean video.
            with torch.no_grad():
                pseudo_tracks, pseudo_visibility = ensemble_predictor(
                    video_clean, queries, only_verifier=True, use_amp=True)["verifier"]
                # pseudo_tracks:     (B, T, N, 2)
                # pseudo_visibility: (B, T, N)

            # Scale pseudo-GT tracks to model input size
            tracks_gt = pseudo_tracks.clone()
            tracks_gt[..., 0] *= (W_model / W)
            tracks_gt[..., 1] *= (H_model / H)

            # Forward pass on augmented video
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                out = model(video_aug, queries_scaled, p_mask=0.1)
                L_p, L_p2, L_vis, L_o, L_u, L_topk_u, L_topk_rank = loss_fn(
                    out, tracks_gt, pseudo_visibility, query_times)

            # Only patch classification + offset + top-k losses (no vis/uncertainty); scale by real weight
            loss = real_weight * (L_p + L_p2 + L_o + L_topk_rank + L_topk_u)

        # === Backward pass ===
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

        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        loss_tracker.update(L_p, L_p2, L_vis, L_o, L_u, L_topk_u, L_topk_rank)

        if args.rank == 0 and use_schedule:
            # Log weights before loss_tracker.log() so they share the same commit=True step
            wandb.log({"Training/syn_weight": syn_weight, "Training/real_weight": real_weight}, commit=False)

        loss_tracker.log()

        desc = f"Loss: {loss.item():.5f}"
        if use_schedule:
            desc += f" | syn_w: {syn_weight:.3f} | real_w: {real_weight:.3f}"
        train_dataloader.set_description(desc)


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

        pred_trajectory = out["P"][:, :, :N]                                       # (1, T, N, P)
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

        # maximum 50 videos for evaluation to limit time
        if j == 59:
            break
        
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
    train_dataloader, val_dataloader = get_dataloaders(args, train_set="real")
    if train_dataloader is not None:
        print(f"Total number of iterations: {len(train_dataloader) * args.epoch_num / 1000:.1f}K")
    # === === ===
    
    # === Model & Training ===
    trackon2_args = load_args_from_yaml(args.trackon2_config_path)
    model = Track_On2(trackon2_args)
    load_pretrained_weights(model, args.trackon2_checkpoint_path)

    model = model.to(args.gpu)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    # # freeze prediction_head.vis_layer and  prediction_head.unc_layer
    # for name, param in model.named_parameters():
    #     if "prediction_head.vis_layer" in name or "prediction_head.unc_layer" in name:
    #         param.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = get_scheduler(args, optimizer, train_dataloader, warmup=False)
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

    # === Load Ensemble Models + Verifier ===
    print("Loading ensemble models...")
    ensemble_models = prepare_models(args)
    ensemble_predictor = Ensemble_Predictor(ensemble_models).to(args.gpu).eval()
    print()
    # === === ===

    # Initial evaluation before training
    if args.rank == 0 and args.checkpoint_path is None:
        # Choose evaluation function based on dataset type
        smaller_delta_avg, aj, oa = evaluate(args, val_dataloader, model, -1)

    print("Training starts")
    for epoch in range(start_epoch, args.epoch_num):
        train_dataloader.sampler.set_epoch(epoch)

        print(f"=== === Epoch {epoch} === ===")

        # === === Training === ===
        total_iterations = len(train_dataloader) * args.epoch_num
        fine_tune(args, train_dataloader, model, ensemble_predictor, optimizer, lr_scheduler, scaler, loss_tracker, epoch, total_iterations)
        print()
        # === === ===
        
        # === Evaluation ===
        if args.rank == 0:
            # Choose evaluation function based on dataset type
            smaller_delta_avg, aj, oa = evaluate(args, val_dataloader, model, epoch)

            # === Update Best Models ===
            best_delta_avg = False
            best_aj = False
            best_oa = False
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
    parser = argparse.ArgumentParser(description="Verifier-Guided Fine-tuning for Track-On2")
    parser.add_argument("--config_path", type=str, required=True, help='Path to the config file')
    args_cmd = parser.parse_args()

    args = load_args_from_yaml(args_cmd.config_path)

    args.gpus = torch.cuda.device_count()
    Path(args.model_save_path).mkdir(parents=True, exist_ok=True)

    main_worker(args)

