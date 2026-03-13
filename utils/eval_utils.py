import os
import pickle
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as Tr

import json
import numpy as np

from typing import Iterable, Mapping, Tuple, Union
import wandb

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
    


def compute_tapvid_metrics(
    query_points: np.ndarray,
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
    query_mode: str,
) -> Mapping[str, np.ndarray]:
    """Computes TAP-Vid metrics (Jaccard, Pts. Within Thresh, Occ. Acc.)
    See the TAP-Vid paper for details on the metric computation.  All inputs are
    given in raster coordinates.  The first three arguments should be the direct
    outputs of the reader: the 'query_points', 'occluded', and 'target_points'.
    The paper metrics assume these are scaled relative to 256x256 images.
    pred_occluded and pred_tracks are your algorithm's predictions.
    This function takes a batch of inputs, and computes metrics separately for
    each video.  The metrics for the full benchmark are a simple mean of the
    metrics across the full set of videos.  These numbers are between 0 and 1,
    but the paper multiplies them by 100 to ease reading.
    Args:
       query_points: The query points, an in the format [t, y, x].  Its size is
         [b, n, 3], where b is the batch size and n is the number of queries
       gt_occluded: A boolean array of shape [b, n, t], where t is the number
         of frames.  True indicates that the point is occluded.
       gt_tracks: The target points, of shape [b, n, t, 2].  Each point is
         in the format [x, y]
       pred_occluded: A boolean array of predicted occlusions, in the same
         format as gt_occluded.
       pred_tracks: An array of track predictions from your algorithm, in the
         same format as gt_tracks.
       query_mode: Either 'first' or 'strided', depending on how queries are
         sampled.  If 'first', we assume the prior knowledge that all points
         before the query point are occluded, and these are removed from the
         evaluation.
    Returns:
        A dict with the following keys:
        occlusion_accuracy: Accuracy at predicting occlusion.
        pts_within_{x} for x in [1, 2, 4, 8, 16]: Fraction of points
          predicted to be within the given pixel threshold, ignoring occlusion
          prediction.
        jaccard_{x} for x in [1, 2, 4, 8, 16]: Jaccard metric for the given
          threshold
        average_pts_within_thresh: average across pts_within_{x}
        average_jaccard: average across jaccard_{x}
    """

    metrics = {}
    # Fixed bug is described in:
    # https://github.com/facebookresearch/co-tracker/issues/20
    eye = np.eye(gt_tracks.shape[2], dtype=np.int32)

    if query_mode == "first":
        # evaluate frames after the query frame
        query_frame_to_eval_frames = np.cumsum(eye, axis=1) - eye
    elif query_mode == "strided":
        # evaluate all frames except the query frame
        query_frame_to_eval_frames = 1 - eye
    else:
        raise ValueError("Unknown query mode " + query_mode)

    query_frame = query_points[..., 0]
    query_frame = np.round(query_frame).astype(np.int32)
    evaluation_points = query_frame_to_eval_frames[query_frame] > 0

    # Occlusion accuracy is simply how often the predicted occlusion equals the
    # ground truth.
    occ_acc = np.sum(
        np.equal(pred_occluded, gt_occluded) & evaluation_points,
        axis=(1, 2),
    ) / np.sum(evaluation_points)
    metrics["occlusion_accuracy"] = occ_acc

    # Next, convert the predictions and ground truth positions into pixel
    # coordinates.
    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(pred_occluded)
    all_frac_within = []
    all_jaccard = []
    for thresh in [1, 2, 4, 8, 16]:
        # True positives are points that are within the threshold and where both
        # the prediction and the ground truth are listed as visible.
        within_dist = np.sum(
            np.square(pred_tracks - gt_tracks),
            axis=-1,
        ) < np.square(thresh)
        is_correct = np.logical_and(within_dist, visible)

        # Compute the frac_within_threshold, which is the fraction of points
        # within the threshold among points that are visible in the ground truth,
        # ignoring whether they're predicted to be visible.
        count_correct = np.sum(
            is_correct & evaluation_points,
            axis=(1, 2),
        )
        count_visible_points = np.sum(visible & evaluation_points, axis=(1, 2))
        frac_correct = count_correct / count_visible_points
        metrics["pts_within_" + str(thresh)] = frac_correct
        all_frac_within.append(frac_correct)

        true_positives = np.sum(
            is_correct & pred_visible & evaluation_points, axis=(1, 2)
        )

        # The denominator of the jaccard metric is the true positives plus
        # false positives plus false negatives.  However, note that true positives
        # plus false negatives is simply the number of points in the ground truth
        # which is easier to compute than trying to compute all three quantities.
        # Thus we just add the number of points in the ground truth to the number
        # of false positives.
        #
        # False positives are simply points that are predicted to be visible,
        # but the ground truth is not visible or too far from the prediction.
        gt_positives = np.sum(visible & evaluation_points, axis=(1, 2))
        false_positives = (~visible) & pred_visible
        false_positives = false_positives | ((~within_dist) & pred_visible)
        false_positives = np.sum(false_positives & evaluation_points, axis=(1, 2))
        jaccard = true_positives / (gt_positives + false_positives)
        metrics["jaccard_" + str(thresh)] = jaccard
        all_jaccard.append(jaccard)
    metrics["average_jaccard"] = np.mean(
        np.stack(all_jaccard, axis=1),
        axis=1,
    )
    metrics["average_pts_within_thresh"] = np.mean(
        np.stack(all_frac_within, axis=1),
        axis=1,
    )
    return metrics


def compute_dr_metrics(pred_trajectory, pred_visibility, trajectory, visibility, H, W, device):
    """Calculate all Dynamic Replica metrics for a given prediction"""
    B, T, N = visibility.shape
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
    
    return out_metrics


def compute_po_metrics(pred_trajectory, pred_visibility, trajectory, visibility, valids, H, W, device):
    """Calculate all Point Odyssey metrics for a given prediction"""
    B, T, N, _ = trajectory.shape
    out_metrics = {}
    
    d_sum = 0.0
    d_vis_sum = 0.0
    thrs = [1, 2, 4, 8, 16]
    sx_ = W / 256.0
    sy_ = H / 256.0
    sc_py = np.array([sx_, sy_]).reshape([1, 1, 2])
    sc_pt = torch.from_numpy(sc_py).float().to(device)
    __, first_visible_inds = torch.max(visibility, dim=1)
    frame_ids_tensor = torch.arange(T, device=device)[None, :, None].repeat(B, 1, N)
    start_tracking_mask = frame_ids_tensor > (first_visible_inds.unsqueeze(1))
    
    for thr in thrs:
        # note we exclude timestep0 from this eval
        d__ = (torch.norm(pred_trajectory[:, 1:] / sc_pt - trajectory[:, 1:] / sc_pt, dim=-1) < thr).float()  # B,S-1,N
        d_ = reduce_masked_mean(d__, valids[:, 1:]).item() * 100.0
        d_sum += d_
        out_metrics['d_%d' % thr] = d_

        d_vis = reduce_masked_mean(d__, visibility[:, 1:] * start_tracking_mask[:, 1:]).item() * 100.0
        d_vis_sum += d_vis
        out_metrics[f"d_vis_{thr}"] = d_vis
        
    d_avg = d_sum / len(thrs)
    d_vis_avg = d_vis_sum / len(thrs)
    out_metrics['d_avg'] = d_avg
    out_metrics['d_vis_avg'] = d_vis_avg
    
    sur_thr = 50
    dists = torch.norm(pred_trajectory / sc_pt - trajectory / sc_pt, dim=-1)  # B,S,N
    dist_ok = 1 - (dists > sur_thr).float() * visibility  # B,S,N
    
    survival = torch.cumprod(dist_ok, dim=1)  # B,S,N
    out_metrics['survival'] = torch.mean(survival).item() * 100.0
    
    # get the median l2 error for each trajectory
    dists_ = dists.permute(0, 2, 1).reshape(B * N, T)
    valids_ = valids.permute(0, 2, 1).reshape(B * N, T)
    visibs_ = visibility.permute(0, 2, 1).reshape(B * N, T)
    median_l2 = reduce_masked_median(dists_, valids_, keep_batch=True)
    median_vis_l2 = reduce_masked_median(dists_, visibs_, keep_batch=True)
    out_metrics['median_l2'] = median_l2.mean().item()
    out_metrics['median_vis_l2'] = median_vis_l2.mean().item()
    
    return out_metrics


def compute_ego_points_metrics(pred_trajectory, pred_visibility, trajectory, visibility, valids, vis_valids, out_of_view, occluded, H, W, device):
    """Calculate all ego_points metrics for a given prediction"""
    B, T, N, _ = trajectory.shape
    out_metrics = {}
    
    d_avg_thrs = [1.0, 2.0, 4.0, 8.0, 16.0]
    d_avg_star_thrs = [8.0, 16.0, 24.0]
    
    # Scaling factors - normalize to 256 like other datasets
    sx_ = W / 256.0
    sy_ = H / 256.0
    sc_py = np.array([sx_, sy_]).reshape([1, 1, 2])
    sc_pt = torch.from_numpy(sc_py).float().to(device)
    
    # Calculate d_avg metrics (exclude first frame)
    for thrs_type, thrs in zip(["old", "new"], [d_avg_thrs, d_avg_star_thrs]):
        d_sum_all = 0.0
        for thr in thrs:
            d_ = (
                torch.norm(pred_trajectory[:, 1:] / sc_pt - trajectory[:, 1:] / sc_pt, dim=-1) < thr
            ).float()  # (B, T-1, N)
            d_all = reduce_masked_mean(d_, valids[:, 1:]).item() * 100.0
            d_sum_all += d_all
            out_metrics[f"{thrs_type}_d_all_{thr}"] = d_all
        
        out_metrics[f"{thrs_type}_d_all_avg"] = d_sum_all / len(thrs)
    
    # Calculate median L2
    dists = torch.norm(pred_trajectory / sc_pt - trajectory / sc_pt, dim=-1)  # (B, T, N)
    dists_ = dists.permute(0, 2, 1).reshape(B * N, T)
    valids_ = valids.permute(0, 2, 1).reshape(B * N, T)
    median_l2 = reduce_masked_median(dists_, valids_, keep_batch=True)
    out_metrics["median_l2"] = median_l2.mean().item()
    
    # Calculate visibility accuracy
    # Handle pred_visibility as logits, probabilities or booleans
    if pred_visibility.dtype == torch.bool:
        pred_vis_binary = pred_visibility.float()
    else:
        pmin = pred_visibility.min()
        pmax = pred_visibility.max()
        # if already in [0,1] treat as probabilities/binary
        if pmin >= 0.0 and pmax <= 1.0:
            pred_vis_binary = pred_visibility.round().float()
        else:
            # assume logits
            pred_vis_binary = torch.sigmoid(pred_visibility).round().float()

    vis_acc_before = (pred_vis_binary == visibility).float()
    vis_acc = reduce_masked_mean(vis_acc_before, vis_valids.float())
    out_metrics["vis_acc"] = vis_acc.mean().item() * 100.0
    
    # Calculate Out-of-View Accuracy (OOVA)
    out_of_view_pred = torch.logical_not(torch.logical_and(
        torch.logical_and(pred_trajectory[:, 1:, :, 0] >= 0.0, pred_trajectory[:, 1:, :, 0] <= W),
        torch.logical_and(pred_trajectory[:, 1:, :, 1] >= 0.0, pred_trajectory[:, 1:, :, 1] <= H)
    )).float()
    
    total_out_of_view = torch.sum(out_of_view[:, 1:])
    if total_out_of_view > 0:
        total_out_of_view_pred = torch.sum(out_of_view[:, 1:] * out_of_view_pred)
        out_metrics["out_of_view"] = (total_out_of_view_pred / total_out_of_view).item() * 100.0
    
    # Calculate In-View Accuracy (IVA)
    in_view_pred = torch.logical_and(
        torch.logical_and(pred_trajectory[:, 1:, :, 0] >= 0.0, pred_trajectory[:, 1:, :, 0] <= W),
        torch.logical_and(pred_trajectory[:, 1:, :, 1] >= 0.0, pred_trajectory[:, 1:, :, 1] <= H)
    ).float()
    
    total_in_view = torch.sum(valids[:, 1:] + occluded[:, 1:])
    if total_in_view > 0:
        total_in_view_pred = torch.sum((valids[:, 1:] + occluded[:, 1:]) * in_view_pred)
        out_metrics["in_view"] = (total_in_view_pred / total_in_view).item() * 100.0
    
    # Calculate Re-ID accuracy
    if total_out_of_view > 0:
        test = out_of_view[:, 1:] * out_of_view_pred
        re_id_perc_sum = 0.0
        for thr in d_avg_star_thrs:
            re_id_points_within = valids[:, 1:] * (torch.norm(pred_trajectory[:, 1:] - trajectory[:, 1:], dim=-1) < thr).float()
            re_id_perc_sum += reduce_masked_mean(re_id_points_within, test).item() * 100.0
        out_metrics["RE_ID_acc"] = re_id_perc_sum / len(d_avg_star_thrs)
    
    return out_metrics



class Evaluator():
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.aj = []
        self.delta_avg = []
        self.oa = []
        self.delta_1 = []
        self.delta_2 = []
        self.delta_4 = []
        self.delta_8 = []
        self.delta_16 = []
        self.cnt = 0

    def get_results(self, epoch, log_to_wandb=True, prefix="Evaluation"):
        results = {
            "delta_avg": sum(self.delta_avg) / len(self.delta_avg),
            "delta_1": sum(self.delta_1) / len(self.delta_1),
            "delta_2": sum(self.delta_2) / len(self.delta_2),
            "delta_4": sum(self.delta_4) / len(self.delta_4),
            "delta_8": sum(self.delta_8) / len(self.delta_8),
            "delta_16": sum(self.delta_16) / len(self.delta_16),
            "aj": sum(self.aj) / len(self.aj),
            "oa": sum(self.oa) / len(self.oa),
        }
        if log_to_wandb:
            wandb.log({f"{prefix}/delta_avg": results["delta_avg"],
                            f"{prefix}/delta_1": results["delta_1"],
                            f"{prefix}/delta_2": results["delta_2"],
                            f"{prefix}/delta_4": results["delta_4"],
                            f"{prefix}/delta_8": results["delta_8"],
                            f"{prefix}/delta_16": results["delta_16"],
                            f"{prefix}/AJ": results["aj"],
                            f"{prefix}/OA": results["oa"],
                            "epoch": epoch}, commit=True)

        print(f"delta_avg: {results['delta_avg']:.2f}")
        print(f"delta_1: {results['delta_1']:.2f}")
        print(f"delta_2: {results['delta_2']:.2f}")
        print(f"delta_4: {results['delta_4']:.2f}")
        print(f"delta_8: {results['delta_8']:.2f}")
        print(f"delta_16: {results['delta_16']:.2f}")
        print(f"AJ: {results['aj']:.2f}")
        print(f"OA: {results['oa']:.2f}")

        return results
        

    def update(self, out_metrics, verbose=False):
        aj = out_metrics['average_jaccard'][0] * 100
        delta = out_metrics['average_pts_within_thresh'][0] * 100
        delta_1 = out_metrics['pts_within_1'][0] * 100
        delta_2 = out_metrics['pts_within_2'][0] * 100
        delta_4 = out_metrics['pts_within_4'][0] * 100
        delta_8 = out_metrics['pts_within_8'][0] * 100
        delta_16 = out_metrics['pts_within_16'][0] * 100
        oa = out_metrics['occlusion_accuracy'][0] * 100

        if verbose:
          print(f"Video {self.cnt} | AJ: {aj:.2f}, delta_avg: {delta:.2f}, OA: {oa:.2f}")
        self.cnt += 1
        
        self.aj.append(aj)
        self.delta_avg.append(delta)
        self.oa.append(oa)
        self.delta_1.append(delta_1)
        self.delta_2.append(delta_2)
        self.delta_4.append(delta_4)
        self.delta_8.append(delta_8)
        self.delta_16.append(delta_16)

    def report(self):
        print(f"Mean AJ: {sum(self.aj) / len(self.aj):.1f}")
        print(f"Mean delta_avg: {sum(self.delta_avg) / len(self.delta_avg):.1f}")
        print(f"Mean delta_1: {sum(self.delta_1) / len(self.delta_1):.1f}")
        print(f"Mean delta_2: {sum(self.delta_2) / len(self.delta_2):.1f}")
        print(f"Mean delta_4: {sum(self.delta_4) / len(self.delta_4):.1f}")
        print(f"Mean delta_8: {sum(self.delta_8) / len(self.delta_8):.1f}")
        print(f"Mean delta_16: {sum(self.delta_16) / len(self.delta_16):.1f}")
        print(f"Mean OA: {sum(self.oa) / len(self.oa):.1f}")
        




class TAPVidEvaluator:
    """Evaluator class for TAP-Vid metrics - supports multiple models"""
    def __init__(self, model_names):
        self.model_names = model_names
        self.model_metrics = {name: {'aj': [], 'delta_avg': [], 'oa': [], 
                                      'delta_1': [], 'delta_2': [], 'delta_4': [], 
                                      'delta_8': [], 'delta_16': []} 
                             for name in model_names}
        self.sample_count = 0
    
    def update(self, model_metrics_dict):
        """Update metrics for all models
        Args:
            model_metrics_dict: dict mapping model_name -> metrics_dict
        """
        for model_name, metrics in model_metrics_dict.items():
            aj = metrics['average_jaccard'][0] * 100
            delta_avg = metrics['average_pts_within_thresh'][0] * 100
            oa = metrics['occlusion_accuracy'][0] * 100
            delta_1 = metrics['pts_within_1'][0] * 100
            delta_2 = metrics['pts_within_2'][0] * 100
            delta_4 = metrics['pts_within_4'][0] * 100
            delta_8 = metrics['pts_within_8'][0] * 100
            delta_16 = metrics['pts_within_16'][0] * 100
            
            self.model_metrics[model_name]['aj'].append(aj)
            self.model_metrics[model_name]['delta_avg'].append(delta_avg)
            self.model_metrics[model_name]['oa'].append(oa)
            self.model_metrics[model_name]['delta_1'].append(delta_1)
            self.model_metrics[model_name]['delta_2'].append(delta_2)
            self.model_metrics[model_name]['delta_4'].append(delta_4)
            self.model_metrics[model_name]['delta_8'].append(delta_8)
            self.model_metrics[model_name]['delta_16'].append(delta_16)
        
        self.sample_count += 1
    
    def get_sample_summary(self, sample_idx, total_samples):
        """Get formatted summary for current sample"""
        lines = [f"\n=== Sample {sample_idx} / {total_samples} ==="]
        lines.append("model name: AJ - delta - OA")
        
        # Find max model name length for alignment
        max_len = max(len(name) for name in self.model_names)
        
        for model_name in self.model_names:
            metrics = self.model_metrics[model_name]
            aj = metrics['aj'][-1] if metrics['aj'] else 0
            delta = metrics['delta_avg'][-1] if metrics['delta_avg'] else 0
            oa = metrics['oa'][-1] if metrics['oa'] else 0
            padded_name = model_name.ljust(max_len)
            lines.append(f"{padded_name}: {aj:.2f} - {delta:.2f} - {oa:.2f}")
        
        lines.append("=== === ===\n")
        return "\n".join(lines)
    
    def get_results(self, step=-1, log_to_wandb=False):
        """Get and print final results for all models"""
        if self.sample_count == 0:
            print("No metrics to report")
            return {}
        
        all_results = {}
        
        print("\n" + "="*60)
        print("FINAL RESULTS - TAP-Vid Metrics")
        print("="*60)
        
        for model_name in self.model_names:
            metrics = self.model_metrics[model_name]
            results = {
                'aj': sum(metrics['aj']) / len(metrics['aj']),
                'delta_avg': sum(metrics['delta_avg']) / len(metrics['delta_avg']),
                'oa': sum(metrics['oa']) / len(metrics['oa']),
                'delta_1': sum(metrics['delta_1']) / len(metrics['delta_1']),
                'delta_2': sum(metrics['delta_2']) / len(metrics['delta_2']),
                'delta_4': sum(metrics['delta_4']) / len(metrics['delta_4']),
                'delta_8': sum(metrics['delta_8']) / len(metrics['delta_8']),
                'delta_16': sum(metrics['delta_16']) / len(metrics['delta_16']),
            }
            all_results[model_name] = results
            
            print(f"\n--- {model_name} ---")
            print(f"AJ: {results['aj']:.2f}")
            print(f"delta_avg: {results['delta_avg']:.2f}")
            print(f"OA: {results['oa']:.2f}")
            print(f"delta_1: {results['delta_1']:.2f}")
            print(f"delta_2: {results['delta_2']:.2f}")
            print(f"delta_4: {results['delta_4']:.2f}")
            print(f"delta_8: {results['delta_8']:.2f}")
            print(f"delta_16: {results['delta_16']:.2f}")
        
        print("\n" + "="*60 + "\n")
        return all_results


class DREvaluator:
    """Evaluator class for Dynamic Replica metrics - supports multiple models"""
    def __init__(self, model_names):
        self.model_names = model_names
        self.model_metrics = {name: {} for name in model_names}
        self.sample_count = 0
    
    def update(self, model_metrics_dict):
        """Update metrics for all models
        Args:
            model_metrics_dict: dict mapping model_name -> metrics_dict
        """
        for model_name, metrics in model_metrics_dict.items():
            for key, value in metrics.items():
                if key not in self.model_metrics[model_name]:
                    self.model_metrics[model_name][key] = []
                self.model_metrics[model_name][key].append(value)
        
        self.sample_count += 1
    
    def get_sample_summary(self, sample_idx, total_samples):
        """Get formatted summary for current sample"""
        lines = [f"\n=== Sample {sample_idx} / {total_samples} ==="]
        lines.append("model name: delta_avg - delta_vis - delta_occ - survival")
        
        # Find max model name length for alignment
        max_len = max(len(name) for name in self.model_names)
        
        for model_name in self.model_names:
            metrics = self.model_metrics[model_name]
            delta_avg = metrics['accuracy'][-1] if 'accuracy' in metrics and metrics['accuracy'] else 0
            delta_vis = metrics['accuracy_vis'][-1] if 'accuracy_vis' in metrics and metrics['accuracy_vis'] else 0
            delta_occ = metrics['accuracy_occ'][-1] if 'accuracy_occ' in metrics and metrics['accuracy_occ'] else 0
            survival = metrics['survival'][-1] if 'survival' in metrics and metrics['survival'] else 0
            padded_name = model_name.ljust(max_len)
            lines.append(f"{padded_name}: {delta_avg:.2f} - {delta_vis:.2f} - {delta_occ:.2f} - {survival:.2f}")
        
        lines.append("=== === ===\n")
        return "\n".join(lines)
    
    def get_results(self, step=-1, log_to_wandb=False):
        """Get and print final results for all models"""
        if self.sample_count == 0:
            print("No metrics to report")
            return {}
        
        all_results = {}
        
        print("\n" + "="*60)
        print("FINAL RESULTS - Dynamic Replica Metrics")
        print("="*60)
        
        for model_name in self.model_names:
            metrics = self.model_metrics[model_name]
            results = {}
            
            for key, values in metrics.items():
                results[key] = sum(values) / len(values)
            
            all_results[model_name] = results
            
            print(f"\n--- {model_name} ---")
            print(f"delta_avg: {results.get('accuracy', 0):.2f}")
            print(f"delta_vis: {results.get('accuracy_vis', 0):.2f}")
            print(f"delta_occ: {results.get('accuracy_occ', 0):.2f}")
            print(f"survival: {results.get('survival', 0):.2f}")
            
            # Print per-threshold metrics
            for thr in [1, 2, 4, 8, 16]:
                if f'accuracy_{thr}' in results:
                    print(f"accuracy_{thr}: {results[f'accuracy_{thr}']:.2f}")
        
        print("\n" + "="*60 + "\n")
        return all_results


class POEvaluator:
    """Evaluator class for Point Odyssey metrics - supports multiple models"""
    def __init__(self, model_names):
        self.model_names = model_names
        self.model_metrics = {name: {} for name in model_names}
        self.sample_count = 0
    
    def update(self, model_metrics_dict):
        """Update metrics for all models
        Args:
            model_metrics_dict: dict mapping model_name -> metrics_dict
        """
        for model_name, metrics in model_metrics_dict.items():
            for key, value in metrics.items():
                if key not in self.model_metrics[model_name]:
                    self.model_metrics[model_name][key] = []
                self.model_metrics[model_name][key].append(value)
        
        self.sample_count += 1
    
    def get_sample_summary(self, sample_idx, total_samples):
        """Get formatted summary for current sample"""
        lines = [f"\n=== Sample {sample_idx} / {total_samples} ==="]
        lines.append("model name: d_avg - d_vis_avg - survival - median_l2")
        
        # Find max model name length for alignment
        max_len = max(len(name) for name in self.model_names)
        
        for model_name in self.model_names:
            metrics = self.model_metrics[model_name]
            d_avg = metrics['d_avg'][-1] if 'd_avg' in metrics and metrics['d_avg'] else 0
            d_vis_avg = metrics['d_vis_avg'][-1] if 'd_vis_avg' in metrics and metrics['d_vis_avg'] else 0
            survival = metrics['survival'][-1] if 'survival' in metrics and metrics['survival'] else 0
            median_l2 = metrics['median_l2'][-1] if 'median_l2' in metrics and metrics['median_l2'] else 0
            padded_name = model_name.ljust(max_len)
            lines.append(f"{padded_name}: {d_avg:.2f} - {d_vis_avg:.2f} - {survival:.2f} - {median_l2:.2f}")
        
        lines.append("=== === ===\n")
        return "\n".join(lines)
    
    def get_results(self, step=-1, log_to_wandb=False):
        """Get and print final results for all models"""
        if self.sample_count == 0:
            print("No metrics to report")
            return {}
        
        all_results = {}
        
        print("\n" + "="*60)
        print("FINAL RESULTS - Point Odyssey Metrics")
        print("="*60)
        
        for model_name in self.model_names:
            metrics = self.model_metrics[model_name]
            results = {}
            
            for key, values in metrics.items():
                results[key] = sum(values) / len(values)
            
            all_results[model_name] = results
            
            print(f"\n--- {model_name} ---")
            print(f"d_avg: {results.get('d_avg', 0):.2f}")
            print(f"d_vis_avg: {results.get('d_vis_avg', 0):.2f}")
            print(f"survival: {results.get('survival', 0):.2f}")
            print(f"median_l2: {results.get('median_l2', 0):.2f}")
            print(f"median_vis_l2: {results.get('median_vis_l2', 0):.2f}")
        
        print("\n" + "="*60 + "\n")
        return all_results


class EgoPointsEvaluator:
    """Evaluator class for ego_points metrics - supports multiple models"""
    def __init__(self, model_names):
        self.model_names = model_names
        self.model_metrics = {name: {} for name in model_names}
        self.sample_count = 0
    
    def update(self, model_metrics_dict):
        """Update metrics for all models
        Args:
            model_metrics_dict: dict mapping model_name -> metrics_dict
        """
        for model_name, metrics in model_metrics_dict.items():
            for key, value in metrics.items():
                if key not in self.model_metrics[model_name]:
                    self.model_metrics[model_name][key] = []
                self.model_metrics[model_name][key].append(value)
        
        self.sample_count += 1
    
    def get_sample_summary(self, sample_idx, total_samples):
        """Get formatted summary for current sample"""
        lines = [f"\n=== Sample {sample_idx} / {total_samples} ==="]
        lines.append("model name: d_avg - d_avg* - MTE - OA")
        
        # Find max model name length for alignment
        max_len = max(len(name) for name in self.model_names)
        
        for model_name in self.model_names:
            metrics = self.model_metrics[model_name]
            d_avg = metrics['old_d_all_avg'][-1] if 'old_d_all_avg' in metrics and metrics['old_d_all_avg'] else 0
            d_avg_star = metrics['new_d_all_avg'][-1] if 'new_d_all_avg' in metrics and metrics['new_d_all_avg'] else 0
            mte = metrics['median_l2'][-1] if 'median_l2' in metrics and metrics['median_l2'] else 0
            oa = metrics['vis_acc'][-1] if 'vis_acc' in metrics and metrics['vis_acc'] else 0
            padded_name = model_name.ljust(max_len)
            lines.append(f"{padded_name}: {d_avg:.2f} - {d_avg_star:.2f} - {mte:.2f} - {oa:.2f}")
        
        lines.append("=== === ===\n")
        return "\n".join(lines)
    
    def get_results(self, step=-1, log_to_wandb=False):
        """Get and print final results for all models"""
        if self.sample_count == 0:
            print("No metrics to report")
            return {}
        
        all_results = {}
        
        print("\n" + "="*60)
        print("FINAL RESULTS - Ego Points Metrics")
        print("="*60)
        
        for model_name in self.model_names:
            metrics = self.model_metrics[model_name]
            results = {}
            
            for key, values in metrics.items():
                results[key] = sum(values) / len(values)
            
            all_results[model_name] = results
            
            print(f"\n--- {model_name} ---")
            print(f"d_avg: {results.get('old_d_all_avg', 0):.2f}")
            print(f"d_avg*: {results.get('new_d_all_avg', 0):.2f}")
            print(f"MTE: {results.get('median_l2', 0):.2f}")
            print(f"OA: {results.get('vis_acc', 0):.2f}")
            
            # Print optional metrics if available
            if 'out_of_view' in results:
                print(f"OOVA: {results['out_of_view']:.2f}")
            if 'in_view' in results:
                print(f"IVA: {results['in_view']:.2f}")
            if 'RE_ID_acc' in results:
                print(f"Re-ID: {results['RE_ID_acc']:.2f}")
        
        print("\n" + "="*60 + "\n")
        return all_results
