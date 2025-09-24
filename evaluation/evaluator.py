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

import argparse
from utils.coord_utils import get_points_on_a_grid
from utils.eval_utils import Evaluator, compute_tapvid_metrics

EPS = 1e-6

def reduce_masked_mean(input, mask, dim=None, keepdim=False):
    mask = mask.expand_as(input)

    prod = input * mask

    if dim is None:
        numer = torch.sum(prod)
        denom = torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer / (EPS + denom)
    return mean


def reduce_masked_median(x, mask, keep_batch=False):
    # x and mask are the same shape
    assert(x.size() == mask.size())
    device = x.device

    B = list(x.shape)[0]
    x = x.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()

    if keep_batch:
        x = np.reshape(x, [B, -1])
        mask = np.reshape(mask, [B, -1])
        meds = np.zeros([B], np.float32)
        for b in list(range(B)):
            xb = x[b]
            mb = mask[b]
            if np.sum(mb) > 0:
                xb = xb[mb > 0]
                meds[b] = np.median(xb)
            else:
                meds[b] = np.nan
        meds = torch.from_numpy(meds).to(device)
        return meds.float()
    else:
        x = np.reshape(x, [-1])
        mask = np.reshape(mask, [-1])
        if np.sum(mask) > 0:
            x = x[mask > 0]
            med = np.median(x)
        else:
            med = np.nan
        med = np.array([med], np.float32)
        med = torch.from_numpy(med).to(device)
        return med.float()
    
@torch.no_grad()
def evaluate_tapvid(model, dataloader, delta_v, print_each=True):
    evaluator = Evaluator()

    for j, (video, trajectory, visibility, query_points_i) in enumerate(dataloader):
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

        pred_trajectory, pred_visibility = model(video, queries)

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
        if print_each:
            print(f"Video {j}/{len(dataloader)}: AJ: {out_metrics['average_jaccard'][0] * 100:.2f}, delta_avg: {out_metrics['average_pts_within_thresh'][0] * 100:.2f}, OA: {out_metrics['occlusion_accuracy'][0] * 100:.2f}", flush=True)
        evaluator.update(out_metrics)

    evaluator.get_results(-1, log_to_wandb=False)

@torch.no_grad()
def evaluate_dr(model, dataloader, print_each=True):
    metrics = {}
    with torch.no_grad():
        for ind, (video, traj_2d, visibility) in enumerate(dataloader):
            trajectory = traj_2d.cuda(non_blocking=True)              # (1, T, N, 2)
            visibility = visibility.cuda(non_blocking=True).float()           # (1, T, N)
            video = video.cuda(non_blocking=True)                     # (1, T, 3, H, W)

            B, T, N, _ = trajectory.shape
            _, _, _, H, W = video.shape

            device = video.device
            queries = torch.cat([torch.zeros_like(trajectory[:, 0, :, :1]), trajectory[:, 0]], dim=2).to(device)
            seq_name = str(ind)
        
            pred_trajectory, pred_visibility = model(video, queries)
            
            # Evaluate
            *_, N, _ = trajectory.shape
            B, T, N = visibility.shape
            H, W = video.shape[-2:]
            device = video.device

            out_metrics = {}

            d_vis_sum = d_occ_sum = d_sum_all = 0.0
            thrs = [1, 2, 4, 8, 16]
            sx_ = (W - 1) / 255.0
            sy_ = (H - 1) / 255.0
            sc_py = np.array([sx_, sy_]).reshape([1, 1, 2])
            sc_pt = torch.from_numpy(sc_py).float().to(device)
            __, first_visible_inds = torch.max(visibility, dim=1)

            frame_ids_tensor = torch.arange(T, device=device)[None, :, None].repeat(B, 1, N)
            start_tracking_mask = frame_ids_tensor > (first_visible_inds.unsqueeze(1))

            for thr in thrs:
                d_ = (
                    torch.norm(
                        pred_trajectory[..., :2] / sc_pt
                        - trajectory[..., :2] / sc_pt,
                        dim=-1,
                    )
                    < thr
                ).float()  # B,S-1,N
                d_occ = (
                    reduce_masked_mean(
                        d_, (1 - visibility) * start_tracking_mask
                    ).item()
                    * 100.0
                )
                d_occ_sum += d_occ
                out_metrics[f"accuracy_occ_{thr}"] = d_occ

                d_vis = (
                    reduce_masked_mean(
                        d_, visibility * start_tracking_mask
                    ).item()
                    * 100.0
                )
                d_vis_sum += d_vis
                out_metrics[f"accuracy_vis_{thr}"] = d_vis

                d_all = reduce_masked_mean(d_, start_tracking_mask).item() * 100.0
                d_sum_all += d_all
                out_metrics[f"accuracy_{thr}"] = d_all

            d_occ_avg = d_occ_sum / len(thrs)
            d_vis_avg = d_vis_sum / len(thrs)
            d_all_avg = d_sum_all / len(thrs)

            sur_thr = 50
            dists = torch.norm(
                pred_trajectory[..., :2] / sc_pt - trajectory[..., :2] / sc_pt,
                dim=-1,
            )  # B,S,N
            dist_ok = 1 - (dists > sur_thr).float() * visibility  # B,S,N
            survival = torch.cumprod(dist_ok, dim=1)  # B,S,N
            out_metrics["survival"] = torch.mean(survival).item() * 100.0

            out_metrics["accuracy_occ"] = d_occ_avg
            out_metrics["accuracy_vis"] = d_vis_avg
            out_metrics["accuracy"] = d_all_avg

            metrics[seq_name] = out_metrics
            for metric_name in out_metrics.keys():
                if "avg" not in metrics:
                    metrics["avg"] = {}
                metrics["avg"][metric_name] = float(
                    np.mean([v[metric_name] for k, v in metrics.items() if k != "avg"])
                )

            if print_each:
                print(f"Sequence {ind}/{len(dataloader)}: delta_avg: {out_metrics['accuracy']:.2f}, delta_vis: {out_metrics['accuracy_vis']:.2f}, delta_occ: {out_metrics['accuracy_occ']:.2f}, survival: {out_metrics['survival']:.2f}", flush=True)

    print()
    for metric_name, value in metrics["avg"].items():
        print(f"{metric_name}: {value:.2f}")
    print()



@torch.no_grad()
def evaluate_po(model, dataloader, print_each=True):
    d_vis_all = []
    d_all = []
    survival_all = []
    median_l2_all = []
    median_vis_l2_all = []
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
            
            trajs_e, pred_visibility = model(video, queries)
            
            metrics = {}
            S = T
            EPS = 1e-6
            d_sum = 0.0
            d_vis_sum = 0.0
            thrs = [1,2,4,8,16]
            sx_ = W / 256.0
            sy_ = H / 256.0
            sc_py = np.array([sx_, sy_]).reshape([1,1,2])
            sc_pt = torch.from_numpy(sc_py).float().cuda()
            __, first_visible_inds = torch.max(visibility, dim=1)
            frame_ids_tensor = torch.arange(T, device=device)[None, :, None].repeat(B, 1, N)
            start_tracking_mask = frame_ids_tensor > (first_visible_inds.unsqueeze(1))
            
            for thr in thrs:
                # note we exclude timestep0 from this eval
                d__ = (torch.norm(trajs_e[:,1:]/sc_pt - trajectory[:,1:]/sc_pt, dim=-1) < thr).float() # B,S-1,N
                d_ = reduce_masked_mean(d__, valids[:,1:]).item()*100.0
                d_sum += d_
                metrics['d_%d' % thr] = d_

                d_vis = reduce_masked_mean(d__, visibility[:,1:] * start_tracking_mask[:,1:]).item() * 100.0
                d_vis_sum += d_vis
                metrics[f"d_vis_{thr}"] = d_vis
                
            d_avg = d_sum / len(thrs)
            d_vis_avg = d_vis_sum / len(thrs)
            metrics['d_avg'] = d_avg
            metrics['d_vis_avg'] = d_vis_avg
            
            sur_thr = 50
            dists = torch.norm(trajs_e/sc_pt - trajectory/sc_pt, dim=-1) # B,S,N
            dist_ok = 1 - (dists > sur_thr).float() * visibility # B,S,N
            
            survival = torch.cumprod(dist_ok, dim=1) # B,S,N
            metrics['survival'] = torch.mean(survival).item()*100.0
            
            # get the median l2 error for each trajectory
            dists_ = dists.permute(0,2,1).reshape(B*N,S)
            valids_ = valids.permute(0,2,1).reshape(B*N,S)
            visibs_ = visibility.permute(0,2,1).reshape(B*N,S)
            median_l2 = reduce_masked_median(dists_, valids_, keep_batch=True)
            median_vis_l2 = reduce_masked_median(dists_, visibs_, keep_batch=True)
            metrics['median_l2'] = median_l2.mean().item()
            metrics['median_vis_l2'] = median_vis_l2.mean().item()

            # print(metrics)
            if print_each:
                print(f"d_avg: {d_avg:.1f}    d_vis_avg: {d_vis_avg:.1f}    survival: {torch.mean(survival).item()*100.0:.1f}    median_l2: {median_l2.mean().item():.1f}    median_vis_l2: {median_vis_l2.mean().item():.1f}", flush=True)
            
            survival_all.append(metrics['survival'])
            d_vis_all.append(metrics['d_vis_avg'])
            d_all.append(metrics['d_avg'])
            median_l2_all.append(metrics['median_l2'])
            median_vis_l2_all.append(metrics['median_vis_l2'])

    print()
    print(f"d_avg: {sum(d_all) / len(d_all)}")   
    print(f"d_vis_avg: {sum(d_vis_all) / len(d_vis_all)}")
    print(f"survival: {sum(survival_all) / len(survival_all)}")
    print(f"median_l2: {sum(median_l2_all) / len(median_l2_all)}")
    print(f"median_vis_l2: {sum(median_vis_l2_all) / len(median_vis_l2_all)}")

