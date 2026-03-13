import os
import sys
import numpy as np
from tqdm import tqdm
import time
import datetime
import math
import copy

from ensemble.locotrack.locotrack_predictor import LocoTrackPredictor
import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import gc
import wandb

from pathlib import Path

import argparse

from utils.train_utils import init_distributed_mode, fix_random_seeds
from utils.train_utils import get_dataloaders, get_scheduler
from utils.train_utils import restart_from_checkpoint, save_on_master
from utils.train_utils import load_args_from_yaml

from utils.log_utils import init_wandb_verifier as init_wandb

from utils.coord_utils import get_queries
from utils.eval_utils import Evaluator, compute_tapvid_metrics

from verifier.verifier import Verifier
from verifier.utils import get_ensemble, select_best_predictions_and_indices, select_diverse_tracks, psi_perturbation

from ensemble.cotracker import CoTracker_Predictor
from ensemble.bootstapir.bootstapir_predictor import TAPIRPredictor
from ensemble.tapnext.tapnext_predictor import TAPNextPredictor
from model.trackon_predictor import Predictor as Trackon_Predictor
from ensemble.locotrack.locotrack_predictor import LocoTrackPredictor
from ensemble.ensemble_predictor import Ensemble_Predictor

def pretrain(args, train_dataloader, verifier, ensemble_predictor, optimizer, lr_scheduler, scaler, epoch):
    verifier.train()
    total_rank_loss = 0.0

    train_dataloader = tqdm(train_dataloader, disable=args.rank != 0, file=sys.stdout)
    update_num = 0

    for i, (video, tracks, visibility) in enumerate(train_dataloader):
        video = video.cuda(non_blocking=True)               # (B, T, 3, H, W)
        tracks = tracks.cuda(non_blocking=True)             # (B, T, N, 2)
        visibility = visibility.bool().cuda(non_blocking=True)       # (B, T, N)

        B, T, N, _ = tracks.shape

        # === Scaling ===
        H_in, W_in = video.shape[-2], video.shape[-1]
        H, W = args.input_size[0], args.input_size[1]

        device = video.device
        # === === ===

        queries = get_queries(tracks, visibility)             # (B, N, 3)
        query_times = queries[:, :, 0].long()               # (B, N)

        time_indices = torch.arange(T).reshape(1, T, 1).to(device)                      # (1, T, 1)
        query_times_expanded = query_times.unsqueeze(1)                                 # (B, 1, N)
        time_mask = (time_indices > query_times_expanded).float()                       # (B, T, N)


        # Use perturbation-based augmentation (original approach)
        M = np.random.choice([6, 8, 10, 12])
        ensemble_tracks = psi_perturbation(tracks.clone(), visibility.clone(), M=M)      # (B, T, N, M, 2)
            
        # Shapes:
        # tracks            : [B, T, N, 2]        (GT)
        # validity          : [B, T, N]           (1 = valid)
        # ensemble_tracks   : [B, T, N, M, 2]     (M trackers’ hypos)
        B, T, N, M, _ = ensemble_tracks.shape

        best_indices, best_predictions = select_best_predictions_and_indices(ensemble_tracks, tracks)   # (B, T, N), (B, T, N, 2)
        err_2d = ensemble_tracks - tracks.unsqueeze(3)          # (B, T, N, M, 2)
        err = torch.norm(err_2d, dim=-1)                        # (B, T, N, M)

        # GT for ranking head
        Tsoft = 0.3
        q = torch.softmax(-(err / Tsoft), dim=-1)                           # (B, T, N, M)
        vis_sampled_mask = visibility.float() * time_mask.float()           # (B, T, N)

        # 5. Verifier output:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
            rank_logits = verifier(video.clone(), 
                                    ensemble_tracks.clone(), 
                                    queries.clone())             # (B, T, N, M), (B, T, N, M, 2)
            
            # === Ranking loss ===
            target = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-6)                # (B, T, N, M) - soft targets
            L_rank = -(target * F.log_softmax(rank_logits, dim=-1)).sum(dim=-1)     # (B, T, N) - manual soft CE
            L_rank = (L_rank * vis_sampled_mask).sum() / (vis_sampled_mask.sum() + 1e-6)
                
            L_rank = L_rank * args.lambda_rank

        loss = L_rank

        # Backward pass
        if args.amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(verifier.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()
            nn.utils.clip_grad_norm_(verifier.parameters(), 1.0)
            optimizer.step()

        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        try:
            del ensemble_tracks, best_indices, best_predictions
        except NameError:
            pass
        torch.cuda.empty_cache()
        gc.collect()

        # log_batch_loss
        update_num += 1
        total_update_num = epoch * len(train_dataloader) + update_num
        if args.rank == 0:
            train_dataloader.set_description(f"Epoch {epoch} - L_rank: {L_rank.item():.3f}")
            L_rank_item = L_rank.detach().item()

            total_rank_loss += L_rank_item

            wandb.log({"Training/ranking_loss": L_rank_item, "iteration": total_update_num}, commit=False)

    # === log_epoch_loss ===
    if args.rank == 0:
        mean_rank_loss = total_rank_loss / (len(train_dataloader))

        wandb.log({"Training/epoch_loss": mean_rank_loss, "epoch": epoch}, commit=False)
        print(f"Mean rank loss: {mean_rank_loss:.3f}\n")
        print()


def _build_ensemble_cache(args, val_dataloader, ensemble_predictor, cache_path):
	"""
	Compute ensemble predictions for the entire validation set and save them to cache_path.
	Only rank 0 computes and writes the cache file; all ranks wait and then load it.
	Stored entries: dict per-sample with keys 'ensemble_tracks','ensemble_visibility','queries' (all CPU tensors).
	"""
	# If cache already exists, load and return immediately (all ranks)
	if os.path.exists(cache_path):
		cache_list = torch.load(cache_path, map_location="cpu")
		return cache_list

	# Only rank 0 computes and writes
	if args.rank == 0:
		cache_list = []
		# ensure val_dataloader iteration order is deterministic and non-shuffled
		for i, (video, trajectory, visibility, query_points_i) in enumerate(tqdm(val_dataloader, disable=args.rank != 0, file=sys.stdout)):
			# move input to GPU for teacher inference
			video_gpu = video.cuda(non_blocking=True)
			queries = query_points_i.clone().float()
			queries = torch.stack([queries[:, :, 0], queries[:, :, 2], queries[:, :, 1]], dim=2).to(video_gpu.device)

			with torch.no_grad():
				# run ensemble predictor once (teachers) and store CPU copies
				with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
					out = ensemble_predictor(video_gpu, queries)
					ensemble_tracks, ensemble_visibility = out["ensemble"]

				cache_list.append({
					"ensemble_tracks": ensemble_tracks.cpu(),
					"ensemble_visibility": ensemble_visibility.cpu(),
					"queries": queries.cpu(),  # stored in case useful
				})

		# save cache (atomic write)
		torch.save(cache_list, cache_path)

	# load cache on all ranks
	cache_list = torch.load(cache_path, map_location="cpu")
	return cache_list


def _prepare_batch(video, trajectory, visibility, query_points_i, device):
	"""Move tensors to device and convert queries from (t,y,x) to (t,x,y)."""
	video = video.to(device, non_blocking=True)
	trajectory = trajectory.to(device, non_blocking=True)
	visibility = visibility.to(device, non_blocking=True)
	q = query_points_i.clone().float().to(device)
	queries = torch.stack([q[:, :, 0], q[:, :, 2], q[:, :, 1]], dim=2).to(device)
	return video, trajectory, visibility, queries

def _run_verifier_and_metrics(evaluator, video, trajectory, visibility, queries,
							  ensemble_tracks, ensemble_visibility, verifier, amp_enabled):
	"""Run verifier on inputs, compute selected tracks and update evaluator."""
	with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
		ranks = verifier.module(video.clone(), 
                                ensemble_tracks.clone(), 
                                queries.clone())                                     # (B,T,N,M), (B,T,N,M,2)
          
		selected_tracks = get_ensemble(ensemble_tracks, ranks, max=True)                     # (B,T,N,2)
		pred_visibility_majority = ensemble_visibility.float().mean(dim=-1) > 0.5            # (B,T,N)

	# prepare arrays for metric computation
	traj = trajectory.clone()
	query_points = queries.clone().cpu().numpy()
	gt_tracks = traj.permute(0, 2, 1, 3).cpu().numpy()
	gt_occluded = torch.logical_not(visibility.clone().permute(0, 2, 1)).cpu().numpy()
	pred_occluded = torch.logical_not(pred_visibility_majority.clone().permute(0, 2, 1)).cpu().numpy()
	pred_tracks = selected_tracks.permute(0, 2, 1, 3).cpu().numpy()

	out_metrics_contrastive = compute_tapvid_metrics(query_points, gt_occluded, gt_tracks, pred_occluded, pred_tracks, "first")
	evaluator.update(out_metrics_contrastive)
     
	return ranks, selected_tracks, pred_visibility_majority

def evaluate(args, val_dataloader, verifier, ensemble_predictor, epoch):
    verifier.eval()
    evaluator = Evaluator()

	# prepare cache path
    cache_enabled = getattr(args, "cache_ensemble", False)
    cache_path = os.path.join(args.model_save_path, "ensemble_cache_val.pt")

	# If caching requested, build cache (rank 0 computes & saves) and load it.
    if cache_enabled:
        cache_list = _build_ensemble_cache(args, val_dataloader, ensemble_predictor, cache_path)
    else:
        cache_list = None
            
    with torch.no_grad():
        if cache_list is None:
            # compute ensemble on the fly
            iter_loader = enumerate(tqdm(val_dataloader, disable=args.rank != 0, file=sys.stdout))
            for i, (video, trajectory, visibility, query_points_i) in iter_loader:
                # prepare batch
                # ensure we use the GPU device (matching the verifier) when available
                device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
                video, trajectory, visibility, queries = _prepare_batch(video, trajectory, visibility, query_points_i, device)

                # compute ensemble (teachers) then verifier + metrics via helper
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                    out = ensemble_predictor(video, queries)
                    ensemble_tracks, ensemble_visibility = out["ensemble"]


                ranks, selected_tracks, pred_visibility_majority = _run_verifier_and_metrics(evaluator, video, trajectory, visibility, queries,
                                            ensemble_tracks, ensemble_visibility, verifier, args.amp)

        else:
            # Use cached ensemble outputs. Iterate val_dataloader and cache_list in lock-step.
            for (video, trajectory, visibility, query_points_i), cache_item in zip(val_dataloader, cache_list):
                # ensure we use the GPU device (matching the verifier) when available
                device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
                video, trajectory, visibility, queries = _prepare_batch(video, trajectory, visibility, query_points_i, device)

                # bring cached ensemble to device
                ensemble_tracks = cache_item["ensemble_tracks"].to(device)
                ensemble_visibility = cache_item["ensemble_visibility"].to(device)


                ranks, selected_tracks, pred_visibility_majority = _run_verifier_and_metrics(evaluator, video, trajectory, visibility, queries,
										  ensemble_tracks, ensemble_visibility, verifier, args.amp)

	# Compute final metrics
    print(f"Evaluation Epoch {epoch}:")
     
    metrics = evaluator.get_results(epoch)

    return metrics['delta_avg'], metrics['aj'], metrics['oa']

def prepare_models(args):
    models = {}

    # === Pre-trained Track-On2 ===
    trackon2_args = load_args_from_yaml(args.trackon2_config_path)
    trackon2_args.M_i = 24
    model = Trackon_Predictor(trackon2_args, checkpoint_path=args.trackon2_checkpoint_path, support_grid_size=5).cuda()
    models["trackon2"] = model
    print(f"Loaded Trackon2 model")

    # === TAPNext ===
    # check if there is bootstapnext_checkpoint_path in args, if not, skip loading TAP-Next and print a warning
    if not hasattr(args, "tapnext_checkpoint_path") or args.tapnext_checkpoint_path is None:
        print(f"Warning: No checkpoint path provided for TAP-Next. Skipping loading TAP-Next model.")
    else:
        model = TAPNextPredictor(args.tapnext_checkpoint_path).cuda()
        models["tapnext"] = model
        print(f"Loaded TAP-Next model")

    # === BootsTAPNext ===
    # check if there is bootstapnext_checkpoint_path in args, if not, skip loading TAP-Next and print a warning
    if not hasattr(args, "bootstapnext_checkpoint_path") or args.bootstapnext_checkpoint_path is None:
        print(f"Warning: No checkpoint path provided for BootsTAPNext. Skipping loading BootsTAPNext model.")
    else:
        model = TAPNextPredictor(args.bootstapnext_checkpoint_path).cuda()
        models["bootstapnext"] = model
        print(f"Loaded BootsTAPNext model")

    # === TAPIR ===
    if not hasattr(args, "tapir_checkpoint_path") or args.tapir_checkpoint_path is None:
        print(f"Warning: No checkpoint path provided for TAPIR. Skipping loading TAPIR model.")
    else:
        model = TAPIRPredictor(args.tapir_checkpoint_path).cuda()
        models['tapir'] = model
        print(f"Loaded TAPIR model")

    # === BootsTAPIR ===
    if not hasattr(args, "bootstapir_checkpoint_path") or args.bootstapir_checkpoint_path is None:
        print(f"Warning: No checkpoint path provided for BootsTAPIR. Skipping loading BootsTAPIR model.")
    else:
        model = TAPIRPredictor(args.bootstapir_checkpoint_path).cuda()
        models["bootstapir"] = model
        print(f"Loaded BootsTAPIR model")

    # # === CoTracker3 (Video) ===
    # model = CoTracker_Predictor(windowed=False).cuda()
    # models["cotracker3_video"] = model
    # print(f"Loaded CoTracker3 Video model")

    # === CoTracker3 (Window) ===
    model = CoTracker_Predictor(windowed=True).cuda()
    models["cotracker3_window"] = model
    print(f"Loaded CoTracker3 Windowed model")

    # === LocoTrack ===
    if not hasattr(args, "locotrack_checkpoint_path") or args.locotrack_checkpoint_path is None:
        print(f"Warning: No checkpoint path provided for LocoTrack. Skipping loading LocoTrack model.")
    else:
        model = LocoTrackPredictor(args.locotrack_checkpoint_path).cuda()
        models['locotrack'] = model
        print(f"Loaded LocoTrack model")

    # === Anthro LocoTrack ===
    if not hasattr(args, "anthro_locotrack_checkpoint_path") or args.anthro_locotrack_checkpoint_path is None:
        print(f"Warning: No checkpoint path provided for Anthro LocoTrack. Skipping loading Anthro LocoTrack model.")
    else:
        model = LocoTrackPredictor(args.anthro_locotrack_checkpoint_path).cuda()
        models['anthro_locotrack'] = model
        print(f"Loaded Anthro LocoTrack model")

    return models


def main_worker(args):
    init_distributed_mode(args)
    fix_random_seeds(args.seed)

    start_time = time.time()

    # === Data ===
    train_dataloader, val_dataloader = get_dataloaders(args, train_set="epic_k")
    if train_dataloader is not None:
        print(f"Total number of iterations: {len(train_dataloader) * args.epoch_num / 1000:.1f}K")
    # === === ===
    
    model = Verifier(args).to(args.gpu)
    for param in model.cnn_encoder.parameters():
        param.requires_grad = False

    
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = get_scheduler(args, optimizer, train_dataloader)
    scaler = torch.GradScaler() if args.amp else None
    init_wandb(args)

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 10**6:.2f}M")
    print(f"Number of learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 10**6:.2f}M")
    print()
    # === === ===

    print("Loading ensemble models...")
    ensemble_models = prepare_models(args)
    ensemble_predictor = Ensemble_Predictor(ensemble_models).to(args.gpu).eval()
    # === === ===

    # === Load from checkpoint ===
    to_restore = {"epoch": 0, "iteration": 0, "best_aj": [-1, -1], "best_oa": [-1, -1], "best_delta_avg": [-1, -1]}

    load_checkpoint = args.checkpoint_path is not None
    continue_training = False
    
    if load_checkpoint:
        if not os.path.exists(args.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint_path}")
        
        continue_training = True
        restart_from_checkpoint(remove_module=False,
                                checkpoint_path=args.checkpoint_path, 
                                run_variables=to_restore, 
                                model=model,
                                scaler=scaler,
                                optimizer=optimizer, 
                                scheduler=lr_scheduler)
        print(f"Continuing training from checkpoint {args.checkpoint_path}")
        

    start_epoch = to_restore["epoch"]
    best_models = {
        "aj": to_restore["best_aj"],
        "oa": to_restore["best_oa"],
        "delta_avg": to_restore["best_delta_avg"]
    }
    # === === ===

    # Initial evaluation before training
    if args.rank == 0 and not continue_training:
        smaller_delta_avg, aj, oa = evaluate(args, val_dataloader, model, ensemble_predictor, -1)
        # print(f"Initial Evaluation - AJ: {aj:.3f}, OA: {oa:.3f}, Smaller Delta Avg: {smaller_delta_avg:.3f}")

    # barrier:
    dist.barrier()

    print("Training starts")
    for epoch in range(start_epoch, args.epoch_num):
        train_dataloader.sampler.set_epoch(epoch)

        print(f"=== === Epoch {epoch} === ===")

        # === === Training === ===
        pretrain(args, train_dataloader, model, ensemble_predictor, optimizer, lr_scheduler, scaler, epoch)

        print()
        # === === ===
        
        # === Evaluation ===
        if args.rank == 0:
            smaller_delta_avg, aj, oa = evaluate(args, val_dataloader, model, ensemble_predictor, epoch)

            # === Update Best Models ===
            best_delta_avg = False
            if aj > best_models["aj"][1]:
                best_models["aj"] = [epoch, aj]

            if oa > best_models["oa"][1]:
                best_models["oa"] = [epoch, oa]

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
            }
        
            save_on_master(save_dict, os.path.join(args.model_save_path, "checkpoint.pt"))

            # save if scores are best
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
    parser = argparse.ArgumentParser(description="Verifier Training")
    parser.add_argument("--config_path", type=str, required=True, help='Path to the config file')
    args_cmd = parser.parse_args()

    args = load_args_from_yaml(args_cmd.config_path)

    args.gpus = torch.cuda.device_count()
    Path(args.model_save_path).mkdir(parents=True, exist_ok=True)

    main_worker(args)