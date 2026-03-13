import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as Tr

import json
import numpy as np

from tqdm import tqdm
import cv2
import imageio.v2 as imageio


from utils.eval_utils import DREvaluator, POEvaluator, EgoPointsEvaluator, TAPVidEvaluator
from utils.eval_utils import compute_tapvid_metrics, compute_dr_metrics, compute_po_metrics, compute_ego_points_metrics


def compute_oracle_prediction(ensemble_tracks_list, ensemble_visibility_list, gt_trajectory, gt_visibility):
    """Compute oracle prediction by selecting, for each track and time step, the
    ensemble member with the lowest L2 error to the ground truth.  This gives
    the upper bound on what any selection strategy could achieve.

    Args:
        ensemble_tracks_list: list of M CPU tensors, each (B, T, N, 2)
        ensemble_visibility_list: list of M CPU tensors, each (B, T, N)
        gt_trajectory: (B, T, N, 2) ground truth positions (on any device)
        gt_visibility: (B, T, N) ground truth visibility (1 = visible)

    Returns:
        oracle_tracks: (B, T, N, 2) CPU tensor
        oracle_visibility: (B, T, N) CPU tensor – uses GT visibility
    """
    ensemble_tracks = torch.stack(ensemble_tracks_list, dim=-2).cpu().float()  # (B, T, N, M, 2)
    gt_traj_cpu = gt_trajectory.cpu().float()  # (B, T, N, 2)

    # L2 distance to GT for every model at every (b, t, n)
    gt_exp = gt_traj_cpu.unsqueeze(-2)                              # (B, T, N, 1, 2)
    errors = torch.norm(ensemble_tracks - gt_exp, dim=-1)          # (B, T, N, M)

    # Index of best model per (b, t, n)
    best_idx = torch.argmin(errors, dim=-1)                        # (B, T, N)

    # Gather oracle tracks
    best_idx_exp = best_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, 2)  # (B, T, N, 1, 2)
    oracle_tracks = torch.gather(ensemble_tracks, dim=-2, index=best_idx_exp).squeeze(-2)  # (B, T, N, 2)

    # Oracle visibility = GT visibility (perfect knowledge)
    oracle_visibility = gt_visibility.cpu().clone().float()
    return oracle_tracks, oracle_visibility


def prepare_tapvid_data(trajectory, visibility, query_points_i, device):
    """Prepare TAP-Vid data for model inference
    
    Args:
        video: (B, T, 3, H, W)
        trajectory: (B, T, N, 2)
        visibility: (B, T, N)
        query_points_i: (B, N, 3) in format (t, y, x)
        device: torch device
        
    Returns:
        queries: (B, N, 3) in format (t, x, y)
        gt_tracks: numpy array for evaluation
        gt_occluded: numpy array for evaluation
        query_points: numpy array for evaluation
    """
    # Change (t, y, x) to (t, x, y)
    queries = query_points_i.clone().float()
    queries = torch.stack([queries[:, :, 0], queries[:, :, 2], queries[:, :, 1]], dim=2).to(device)
    
    # Prepare ground truth for evaluation (From CoTracker format)
    traj = trajectory.clone()
    query_points = query_points_i.clone().cpu().numpy()
    gt_tracks = traj.permute(0, 2, 1, 3).cpu().numpy()
    gt_occluded = torch.logical_not(visibility.clone().permute(0, 2, 1)).cpu().numpy()
    
    return queries, gt_tracks, gt_occluded, query_points


@torch.no_grad()
def evaluate_tapvid(models, dataloader, print_each=True, cache_predictions=False, cache_dir=None, dataset_name=None):
    """Evaluate multiple models on TAP-Vid dataset
    
    Args:
        models: dict mapping model_name -> model or single model
        dataloader: TAP-Vid dataloader
        print_each: whether to print per-sample metrics
    """
    # Handle single model or dict of models
    if not isinstance(models, dict):
        models = {'model': models}
    
    # Check if verifier / oracle / random are in the models
    has_verifier = 'verifier' in models
    if has_verifier:
        verifier = models.pop('verifier')  # Remove verifier, will run it last
    has_oracle = 'oracle' in models
    if has_oracle:
        models.pop('oracle')  # Oracle has no model; computed from other predictions
    has_random = 'random' in models
    if has_random:
        models.pop('random')  # Random has no model; computed from other predictions

    model_names = list(models.keys())
    if has_verifier:
        model_names.append('verifier')  # Add verifier to evaluated models
    if has_oracle:
        model_names.append('oracle')  # Oracle upper bound
    if has_random:
        model_names.append('random')  # Round-robin random baseline

    evaluator = TAPVidEvaluator(model_names)

    for j, (video, trajectory, visibility, query_points_i) in enumerate(dataloader):
        query_points_i = query_points_i.cuda(non_blocking=True)      # (1, N, 3)
        trajectory = trajectory.cuda(non_blocking=True)              # (1, T, N, 2)
        visibility = visibility.cuda(non_blocking=True)              # (1, T, N)
        video = video.cuda(non_blocking=True)                    # (1, T, 3, H, W)
        B, T, N, _ = trajectory.shape
        _, _, _, H, W = video.shape
        device = video.device

        # Prepare data for all models
        queries, gt_tracks, gt_occluded, query_points = prepare_tapvid_data(
            trajectory, visibility, query_points_i, device
        )
        
        # Storage for ensemble predictions (if verifier is present)
        ensemble_tracks_list = []
        ensemble_visibility_list = []
        
        # Evaluate each model (except verifier)
        model_metrics = {}
        for model_name, model in models.items():
            # Check cache first
            cache_path = None
            if cache_predictions and cache_dir and dataset_name:
                cache_path = os.path.join(cache_dir, dataset_name, model_name, f"{j:06d}.npz")
                if os.path.exists(cache_path):
                    # Load from cache
                    cached = np.load(cache_path)
                    pred_trajectory = torch.from_numpy(cached['tracks']).to(device)
                    pred_visibility = torch.from_numpy(cached['visibility']).to(device)
                else:
                    # Run model
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        pred_trajectory, pred_visibility = model(video.clone(), queries.clone())
                    # Save to cache
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    np.savez(cache_path, 
                            tracks=pred_trajectory.cpu().numpy(), 
                            visibility=pred_visibility.cpu().numpy())
            else:
                # Run model without caching
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred_trajectory, pred_visibility = model(video.clone(), queries.clone())
            
            # Store predictions for verifier / oracle / random (keep on CPU to save GPU memory)
            if has_verifier or has_oracle or has_random:
                ensemble_tracks_list.append(pred_trajectory.clone().cpu())  # (B, T, N, 2)
                ensemble_visibility_list.append(pred_visibility.clone().cpu())  # (B, T, N)

            # Convert predictions to evaluation format
            pred_occluded = torch.logical_not(pred_visibility.clone().permute(0, 2, 1)).cpu().numpy()
            pred_tracks = pred_trajectory.permute(0, 2, 1, 3).cpu().numpy()
            
            # Compute metrics
            out_metrics = compute_tapvid_metrics(
                query_points, gt_occluded, gt_tracks, pred_occluded, pred_tracks, "first"
            )
            model_metrics[model_name] = out_metrics
        
        # If verifier is present, run it with ensemble predictions
        if has_verifier:
            # Stack ensemble predictions and move to device
            ensemble_tracks = torch.stack(ensemble_tracks_list, dim=-2).to(device)  # (B, T, N, M, 2)
            ensemble_visibility = torch.stack(ensemble_visibility_list, dim=-1).to(device)  # (B, T, N, M)
            
            # Run verifier
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred_trajectory, pred_visibility = verifier(
                    video.clone(), 
                    queries.clone(), 
                    ensemble_tracks, 
                    ensemble_visibility
                )
            
            # Convert predictions to evaluation format
            pred_occluded = torch.logical_not(pred_visibility.clone().permute(0, 2, 1)).cpu().numpy()
            pred_tracks = pred_trajectory.permute(0, 2, 1, 3).cpu().numpy()
            
            # Compute metrics for verifier
            out_metrics = compute_tapvid_metrics(
                query_points, gt_occluded, gt_tracks, pred_occluded, pred_tracks, "first"
            )
            model_metrics['verifier'] = out_metrics

        # Compute oracle: per-track, per-timestep best prediction across all models
        if has_oracle:
            oracle_tracks, oracle_visibility = compute_oracle_prediction(
                ensemble_tracks_list, ensemble_visibility_list, trajectory, visibility
            )
            oracle_tracks = oracle_tracks.to(device)
            oracle_visibility = oracle_visibility.to(device)
            pred_occluded = torch.logical_not(oracle_visibility.permute(0, 2, 1)).cpu().numpy()
            pred_tracks_oracle = oracle_tracks.permute(0, 2, 1, 3).cpu().numpy()
            model_metrics['oracle'] = compute_tapvid_metrics(
                query_points, gt_occluded, gt_tracks, pred_occluded, pred_tracks_oracle, "first"
            )

        # Compute random: round-robin model selection z = j % M
        if has_random:
            M = len(ensemble_tracks_list)
            z = j % M
            rand_tracks = ensemble_tracks_list[z].to(device)
            rand_visibility = ensemble_visibility_list[z].to(device)
            pred_occluded = torch.logical_not(rand_visibility.permute(0, 2, 1)).cpu().numpy()
            pred_tracks_rand = rand_tracks.permute(0, 2, 1, 3).cpu().numpy()
            model_metrics['random'] = compute_tapvid_metrics(
                query_points, gt_occluded, gt_tracks, pred_occluded, pred_tracks_rand, "first"
            )

        # Update evaluator
        evaluator.update(model_metrics)
        
        # Print per-sample summary
        if print_each:
            print(evaluator.get_sample_summary(j + 1, len(dataloader)), flush=True)

    # Print final results
    evaluator.get_results(-1, log_to_wandb=False)


@torch.no_grad()
def evaluate_dr(models, dataloader, print_each=True, cache_predictions=False, cache_dir=None, dataset_name=None):
    """Evaluate multiple models on Dynamic Replica dataset
    
    Args:
        models: dict mapping model_name -> model or single model
        dataloader: Dynamic Replica dataloader
        print_each: whether to print per-sample metrics
    """
    # Handle single model or dict of models
    if not isinstance(models, dict):
        models = {'model': models}
    
    # Check if verifier / oracle / random are in the models
    has_verifier = 'verifier' in models
    if has_verifier:
        verifier = models.pop('verifier')  # Remove verifier, will run it last
    has_oracle = 'oracle' in models
    if has_oracle:
        models.pop('oracle')  # Oracle has no model; computed from other predictions
    has_random = 'random' in models
    if has_random:
        models.pop('random')  # Random has no model; computed from other predictions

    model_names = list(models.keys())
    if has_verifier:
        model_names.append('verifier')  # Add verifier to evaluated models
    if has_oracle:
        model_names.append('oracle')  # Oracle upper bound
    if has_random:
        model_names.append('random')  # Round-robin random baseline

    evaluator = DREvaluator(model_names)
    
    with torch.no_grad():
        for ind, (video, traj_2d, visibility) in enumerate(dataloader):
            trajectory = traj_2d.cuda(non_blocking=True)              # (1, T, N, 2)
            visibility = visibility.cuda(non_blocking=True).float()           # (1, T, N)
            video = video.cuda(non_blocking=True)                     # (1, T, 3, H, W)

            B, T, N, _ = trajectory.shape
            _, _, _, H, W = video.shape

            device = video.device
            queries = torch.cat([torch.zeros_like(trajectory[:, 0, :, :1]), trajectory[:, 0]], dim=2).to(device)
            
            # Storage for ensemble predictions (if verifier is present)
            ensemble_tracks_list = []
            ensemble_visibility_list = []
            
            # Evaluate each model (except verifier)
            model_metrics = {}
            for model_name, model in models.items():
                # Check cache first
                cache_path = None
                if cache_predictions and cache_dir and dataset_name:
                    cache_path = os.path.join(cache_dir, dataset_name, model_name, f"{ind:06d}.npz")
                    if os.path.exists(cache_path):
                        # Load from cache
                        cached = np.load(cache_path)
                        pred_trajectory = torch.from_numpy(cached['tracks']).to(device)
                        pred_visibility = torch.from_numpy(cached['visibility']).to(device)
                    else:
                        # Run model
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            pred_trajectory, pred_visibility = model(video.clone(), queries.clone())
                        # Save to cache
                        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                        np.savez(cache_path, 
                                tracks=pred_trajectory.cpu().numpy(), 
                                visibility=pred_visibility.cpu().numpy())
                else:
                    # Run model without caching
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        pred_trajectory, pred_visibility = model(video.clone(), queries.clone())
                
                # Store predictions for verifier / oracle / random (keep on CPU to save GPU memory)
                if has_verifier or has_oracle or has_random:
                    ensemble_tracks_list.append(pred_trajectory.clone().cpu())  # (B, T, N, 2)
                    ensemble_visibility_list.append(pred_visibility.clone().cpu())  # (B, T, N)


                pred_trajectory = pred_trajectory.to(device)
                pred_visibility = pred_visibility.to(device)
                
                # Calculate metrics
                out_metrics = compute_dr_metrics(pred_trajectory, pred_visibility, trajectory, visibility, H, W, device)
                model_metrics[model_name] = out_metrics
            
            # If verifier is present, run it with ensemble predictions
            if has_verifier:
                # Stack ensemble predictions and move to device
                ensemble_tracks = torch.stack(ensemble_tracks_list, dim=-2).to(device)  # (B, T, N, M, 2)
                ensemble_visibility = torch.stack(ensemble_visibility_list, dim=-1).to(device)  # (B, T, N, M)
                
                # Run verifier
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred_trajectory, pred_visibility = verifier(
                        video.clone(), 
                        queries.clone(), 
                        ensemble_tracks, 
                        ensemble_visibility)
                    
                    pred_trajectory = pred_trajectory.to(device)
                    pred_visibility = pred_visibility.to(device)
                    
                
                # Calculate metrics for verifier
                out_metrics = compute_dr_metrics(pred_trajectory, pred_visibility, trajectory, visibility, H, W, device)
                model_metrics['verifier'] = out_metrics

            # Compute oracle: per-track, per-timestep best prediction across all models
            if has_oracle:
                oracle_tracks, oracle_visibility = compute_oracle_prediction(
                    ensemble_tracks_list, ensemble_visibility_list, trajectory, visibility
                )
                oracle_tracks = oracle_tracks.to(device)
                oracle_visibility = oracle_visibility.to(device)
                model_metrics['oracle'] = compute_dr_metrics(
                    oracle_tracks, oracle_visibility, trajectory, visibility, H, W, device
                )

            # Compute random: round-robin model selection z = ind % M
            if has_random:
                M = len(ensemble_tracks_list)
                z = ind % M
                rand_tracks = ensemble_tracks_list[z].to(device)
                rand_visibility = ensemble_visibility_list[z].to(device)
                model_metrics['random'] = compute_dr_metrics(
                    rand_tracks, rand_visibility, trajectory, visibility, H, W, device
                )

            evaluator.update(model_metrics)

            if print_each:
                print(evaluator.get_sample_summary(ind + 1, len(dataloader)), flush=True)

    evaluator.get_results(-1, log_to_wandb=False)


@torch.no_grad()
def evaluate_po(models, dataloader, print_each=True, cache_predictions=False, cache_dir=None, dataset_name=None):
    """Evaluate multiple models on Point Odyssey dataset
    
    Args:
        models: dict mapping model_name -> model or single model
        dataloader: Point Odyssey dataloader
        print_each: whether to print per-sample metrics
    """
    # Handle single model or dict of models
    if not isinstance(models, dict):
        models = {'model': models}
    
    # Check if verifier / oracle / random are in the models
    has_verifier = 'verifier' in models
    if has_verifier:
        verifier = models.pop('verifier')  # Remove verifier, will run it last
    has_oracle = 'oracle' in models
    if has_oracle:
        models.pop('oracle')  # Oracle has no model; computed from other predictions
    has_random = 'random' in models
    if has_random:
        models.pop('random')  # Random has no model; computed from other predictions

    model_names = list(models.keys())
    if has_verifier:
        model_names.append('verifier')  # Add verifier to evaluated models
    if has_oracle:
        model_names.append('oracle')  # Oracle upper bound
    if has_random:
        model_names.append('random')  # Round-robin random baseline

    evaluator = POEvaluator(model_names)
    
    with torch.no_grad():
        for j, sample in enumerate(dataloader):

            if sample == torch.zeros(1, 1):
                continue

            seq = str(sample['seq'][0])
            print('seq', seq)
        
            trajectory = sample['trajs'].cuda(non_blocking=True).float()              # (1, T, N, 2)
            visibility = sample['visibs'].cuda(non_blocking=True).float()              # (1, T, N)
            valids = sample['valids'].cuda(non_blocking=True)
            

            B, T, N, _ = trajectory.shape
            device = visibility.device

            rgb_path0 = os.path.join(seq, 'rgbs', 'rgb_%05d.jpg' % (0))
            rgb0_bak = cv2.imread(rgb_path0)
            H_bak, W_bak = rgb0_bak.shape[:2]
            H, W = (384,512)
            sy = H/H_bak
            sx = W/W_bak
            trajectory[:,:,:,0] *= sx
            trajectory[:,:,:,1] *= sy

            # read images into tensor
            rgb_paths_seq = [os.path.join(seq, 'rgbs', 'rgb_%05d.jpg' % (idx)) for idx in range(T)]
            rgbs = []
            rgb_paths_seq_iter = tqdm(rgb_paths_seq)
            for rgb_path in rgb_paths_seq_iter:
                rgb_i = cv2.imread(rgb_path)
                rgb_i = rgb_i[:,:,::-1]                                             # BGR->RGB
                rgb_i = cv2.resize(rgb_i, (W, H), interpolation=cv2.INTER_LINEAR)   # resize
                rgbs.append(torch.from_numpy(rgb_i).permute(2, 0, 1).unsqueeze(0))               # H, W, 3 -> 1, 3, H, W

            video = torch.cat(rgbs, dim=0).float().cuda()
            video = video.unsqueeze(dim=0)
            # ===

            # Find queries
            _, first_visible_inds = torch.max(visibility, dim=1)  # (1, N)
            time_indices = first_visible_inds.squeeze(0)   # (N)
            first_visible_points = trajectory[0, time_indices, torch.arange(trajectory.shape[2])] # (N, 2)
            queries = torch.cat([first_visible_inds.unsqueeze(-1), first_visible_points.unsqueeze(0)], dim=2).to(device)  # (1, N, 3)
            
            # Storage for ensemble predictions (if verifier is present)
            ensemble_tracks_list = []
            ensemble_visibility_list = []
            
            # Evaluate each model (except verifier)
            model_metrics = {}
            for model_name, model in models.items():
                # Check cache first
                cache_path = None
                if cache_predictions and cache_dir and dataset_name:
                    cache_path = os.path.join(cache_dir, dataset_name, model_name, f"{j:06d}.npz")
                    if os.path.exists(cache_path):
                        # Load from cache
                        cached = np.load(cache_path)
                        pred_trajectory = torch.from_numpy(cached['tracks']).to(device)
                        pred_visibility = torch.from_numpy(cached['visibility']).to(device)
                    else:
                        # Run model
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            pred_trajectory, pred_visibility = model(video.clone(), queries.clone())
                        # Save to cache
                        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                        np.savez(cache_path, 
                                tracks=pred_trajectory.cpu().numpy(), 
                                visibility=pred_visibility.cpu().numpy())
                else:
                    # Run model without caching
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        pred_trajectory, pred_visibility = model(video.clone(), queries.clone())
                
                # Store predictions for verifier / oracle / random (keep on CPU to save GPU memory)
                if has_verifier or has_oracle or has_random:
                    ensemble_tracks_list.append(pred_trajectory.clone().cpu())  # (B, T, N, 2)
                    ensemble_visibility_list.append(pred_visibility.clone().cpu())  # (B, T, N)

                # Calculate metrics
                pred_trajectory = pred_trajectory.to(device)
                pred_visibility = pred_visibility.to(device)

                out_metrics = compute_po_metrics(pred_trajectory, pred_visibility, trajectory, visibility, valids, H, W, device)
                model_metrics[model_name] = out_metrics
            
            # If verifier is present, run it with ensemble predictions
            if has_verifier:
                # Stack ensemble predictions and move to device
                ensemble_tracks = torch.stack(ensemble_tracks_list, dim=-2).to(device)  # (B, T, N, M, 2)
                ensemble_visibility = torch.stack(ensemble_visibility_list, dim=-1).to(device)  # (B, T, N, M)
                
                # Run verifier
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred_trajectory, pred_visibility = verifier(
                        video.clone(), 
                        queries.clone(), 
                        ensemble_tracks, 
                        ensemble_visibility
                    )

                    pred_trajectory = pred_trajectory.to(device)
                    pred_visibility = pred_visibility.to(device)
                
                # Calculate metrics for verifier
                out_metrics = compute_po_metrics(pred_trajectory, pred_visibility, trajectory, visibility, valids, H, W, device)
                model_metrics['verifier'] = out_metrics

            # Compute oracle: per-track, per-timestep best prediction across all models
            if has_oracle:
                oracle_tracks, oracle_visibility = compute_oracle_prediction(
                    ensemble_tracks_list, ensemble_visibility_list, trajectory, visibility
                )
                oracle_tracks = oracle_tracks.to(device)
                oracle_visibility = oracle_visibility.to(device)
                model_metrics['oracle'] = compute_po_metrics(
                    oracle_tracks, oracle_visibility, trajectory, visibility, valids, H, W, device
                )

            # Compute random: round-robin model selection z = j % M
            if has_random:
                M = len(ensemble_tracks_list)
                z = j % M
                rand_tracks = ensemble_tracks_list[z].to(device)
                rand_visibility = ensemble_visibility_list[z].to(device)
                model_metrics['random'] = compute_po_metrics(
                    rand_tracks, rand_visibility, trajectory, visibility, valids, H, W, device
                )

            evaluator.update(model_metrics)

            if print_each:
                print(evaluator.get_sample_summary(j + 1, len(dataloader)), flush=True)

    evaluator.get_results(-1, log_to_wandb=False)


@torch.no_grad()
def evaluate_ego_points(models, dataloader, print_each=True, cache_predictions=False, cache_dir=None, dataset_name=None):
    """Evaluate multiple models on Ego Points dataset
    
    Args:
        models: dict mapping model_name -> model or single model
        dataloader: Ego Points dataloader
        print_each: whether to print per-sample metrics
    """
    # Handle single model or dict of models
    if not isinstance(models, dict):
        models = {'model': models}
    
    # Check if verifier / oracle / random are in the models
    has_verifier = 'verifier' in models
    if has_verifier:
        verifier = models.pop('verifier')  # Remove verifier, will run it last
    has_oracle = 'oracle' in models
    if has_oracle:
        models.pop('oracle')  # Oracle has no model; computed from other predictions
    has_random = 'random' in models
    if has_random:
        models.pop('random')  # Random has no model; computed from other predictions

    model_names = list(models.keys())
    if has_verifier:
        model_names.append('verifier')  # Add verifier to evaluated models
    if has_oracle:
        model_names.append('oracle')  # Oracle upper bound
    if has_random:
        model_names.append('random')  # Round-robin random baseline

    evaluator = EgoPointsEvaluator(model_names)

    with torch.no_grad():
        for j, sample in enumerate(dataloader):
            trajectory = sample['trajectory'].cuda(non_blocking=True)  # (1, T, N, 2)
            video = sample['video'].cuda(non_blocking=True)  # (1, T, 3, H, W)
            visibility = sample['visibility'].cuda(non_blocking=True)  # (1, T, N)
            valids = sample['valids'].cuda(non_blocking=True)  # (1, T, N)
            vis_valids = sample['vis_valids'].cuda(non_blocking=True)  # (1, T, N)
            out_of_view = sample['out_of_view'].cuda(non_blocking=True)  # (1, T, N)
            occluded = sample['occluded'].cuda(non_blocking=True)  # (1, T, N)
            seq = sample['seq'][0]
            
            B, T, N, _ = trajectory.shape
            _, _, _, H, W = video.shape
            device = video.device
            
            # Create queries from first frame positions (t=0, x, y)
            queries = torch.cat([torch.zeros_like(trajectory[:, 0, :, :1]), trajectory[:, 0]], dim=2).to(device)  # (1, N, 3)

            # Storage for ensemble predictions (if verifier is present)
            ensemble_tracks_list = []
            ensemble_visibility_list = []

            # Evaluate each model (except verifier)
            model_metrics = {}
            for model_name, model in models.items():
                # Check cache first
                cache_path = None
                if cache_predictions and cache_dir and dataset_name:
                    cache_path = os.path.join(cache_dir, dataset_name, model_name, f"{j:06d}.npz")
                    if os.path.exists(cache_path):
                        # Load from cache
                        cached = np.load(cache_path)
                        pred_trajectory = torch.from_numpy(cached['tracks']).to(device)
                        pred_visibility = torch.from_numpy(cached['visibility']).to(device)
                    else:
                        # Run model
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            pred_trajectory, pred_visibility = model(video.clone(), queries.clone())

                        # Save to cache
                        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                        np.savez(cache_path, 
                                tracks=pred_trajectory.cpu().numpy(), 
                                visibility=pred_visibility.cpu().numpy())
                else:
                    # Run model without caching
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        pred_trajectory, pred_visibility = model(video.clone(), queries.clone())

                
                # Store predictions for verifier / oracle / random (keep on CPU to save GPU memory)
                if has_verifier or has_oracle or has_random:
                    ensemble_tracks_list.append(pred_trajectory.clone().cpu())  # (B, T, N, 2)
                    ensemble_visibility_list.append(pred_visibility.clone().cpu())  # (B, T, N)

                pred_trajectory = pred_trajectory.to(device)
                pred_visibility = pred_visibility.to(device)

                # Calculate metrics
                out_metrics = compute_ego_points_metrics(
                    pred_trajectory, pred_visibility, trajectory, visibility, 
                    valids, vis_valids, out_of_view, occluded, H, W, device
                )
                model_metrics[model_name] = out_metrics
            
            # If verifier is present, run it with ensemble predictions
            if has_verifier:
                # Stack ensemble predictions and move to device
                ensemble_tracks = torch.stack(ensemble_tracks_list, dim=-2).to(device)  # (B, T, N, M, 2)
                ensemble_visibility = torch.stack(ensemble_visibility_list, dim=-1).to(device)  # (B, T, N, M)
                
                # Run verifier
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred_trajectory, pred_visibility = verifier(
                        video.clone(), 
                        queries.clone(), 
                        ensemble_tracks, 
                        ensemble_visibility
                    )
                    pred_trajectory = pred_trajectory.to(device)
                    pred_visibility = pred_visibility.to(device)
                
                # Calculate metrics for verifier
                out_metrics = compute_ego_points_metrics(
                    pred_trajectory, pred_visibility, trajectory, visibility, 
                    valids, vis_valids, out_of_view, occluded, H, W, device
                )
                model_metrics['verifier'] = out_metrics

            # Compute oracle: per-track, per-timestep best prediction across all models
            if has_oracle:
                oracle_tracks, oracle_visibility = compute_oracle_prediction(
                    ensemble_tracks_list, ensemble_visibility_list, trajectory, visibility
                )
                oracle_tracks = oracle_tracks.to(device)
                oracle_visibility = oracle_visibility.to(device)
                model_metrics['oracle'] = compute_ego_points_metrics(
                    oracle_tracks, oracle_visibility, trajectory, visibility,
                    valids, vis_valids, out_of_view, occluded, H, W, device
                )

            # Compute random: round-robin model selection z = j % M
            if has_random:
                M = len(ensemble_tracks_list)
                z = j % M
                rand_tracks = ensemble_tracks_list[z].to(device)
                rand_visibility = ensemble_visibility_list[z].to(device)
                model_metrics['random'] = compute_ego_points_metrics(
                    rand_tracks, rand_visibility, trajectory, visibility,
                    valids, vis_valids, out_of_view, occluded, H, W, device
                )

            evaluator.update(model_metrics)
            
            if print_each:
                print(evaluator.get_sample_summary(j + 1, len(dataloader)), flush=True)

    evaluator.get_results(-1, log_to_wandb=False)

